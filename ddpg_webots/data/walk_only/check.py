"""Bounded offline checks: command clock, reference scoring and online-only interfaces."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'runtime'))
sys.path.insert(0,str(OUT.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.learning import create_algorithm,OnlineFIFOReplay
from basewalk.walk_commands import WalkCommands
from basewalk.gradual_turn import source_motion_stage
from basewalk.training_design import update_schedule
from basewalk.phase_fall import PhaseFallBuffer,fall_penalty_at
from basewalk.network_inputs import actor_indices,actor_input_dim,critic_input_dim


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    cfg=training_config(training_configuration(OUT/'config.json'))
    score=load(OUT/'reward.py','walk_score')
    reference=load(OUT/'assets/sac_reference_reward.py','sac_walk_score')
    info=json.loads((OUT/'reference.json').read_text())
    expected=[[.04,0],[.04,.03],[0,.03],[-.03,.03],[-.03,0],[-.03,-.03],[0,-.03],[.04,-.03],
              [.06,0],[.06,-.04],[0,-.04],[-.04,-.04],[-.04,0],[-.04,.04],[0,.04],[.06,.04]]
    clock=WalkCommands(cfg)
    for step in range(3200):
        command=clock.command(0,0)
        np.testing.assert_allclose(command,[*expected[(step//100)%16],0],atol=1e-8)
        assert source_motion_stage(command,cfg,clock.stage)=='walk'
        clock.advance_phase(.9,.1 if step%2 else .95,fell=step%11==0)
        clock=WalkCommands(cfg,state=clock.state())
    assert clock.state()['step']==3200
    for patch in (dict(hold_steps=0),dict(commands_m_s=[[0,0]]),dict(commands_m_s=[[.2,0]])):
        bad=copy.deepcopy(cfg);bad['training']['walk_commands'].update(patch)
        try:WalkCommands(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid commands accepted')
    rng=np.random.default_rng(4)
    max_error=0.
    for command in expected:
        for _ in range(50):
            velocity=rng.normal(0,.08,3);gyro=rng.normal(0,.1,2)
            gravity=np.array([.01,.02,-math.sqrt(1-.01**2-.02**2)])
            a=score.motion_reward([*command,0],velocity,gyro,gravity,2,cfg)[0]
            b=reference.motion_reward([*command,0],velocity,gyro,gravity,2,cfg)[0]
            max_error=max(max_error,abs(a-b))
    assert max_error==0.
    peak=score.motion_reward([.04,.03,0],[.04,.03,0],[0,0],[0,0,-1],2,cfg)[0]
    assert peak==1.25
    assert abs(score.motion_reward([.04,.03,0],[.076,.066,0],[0,0],[0,0,-1],2,cfg)[0]-1.25*math.exp(-2))<1e-12
    sac_fall=load(OUT/'assets/sac_reference_phase_fall.py','reference_fall')
    assert all(fall_penalty_at(x,5)==sac_fall.fall_penalty_at(x,5) for x in np.linspace(.01,1,100))
    agent=create_algorithm(cfg)
    assert isinstance(agent.replay,OnlineFIFOReplay) and agent.replay.count==0 and agent.updates==0
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state
    assert not any(k in cfg for k in ('three_class_online','outcome_replay','expert_pipeline','bc_rollout'))
    assert actor_input_dim(cfg)==50 and critic_input_dim(cfg,0)==61
    assert cfg['training']['learning_starts']==1000
    for step in (999,1000,1001,1019):assert update_schedule(cfg['training'],step,0)==(False,1)
    source=OUT.parents[1]/'expert/step_wide_gaussian/turn_and_walk/expert.pth'
    data=torch.load(source,map_location='cpu',weights_only=False)
    obs=[];nxt=[];modes=[]
    for episode in data['episodes']:
        arrays=episode['arrays'];mask=np.asarray(arrays['modes'])==2
        if mask.any():
            obs.extend(arrays['observations'][mask]);nxt.extend(arrays['next_observations'][mask]);modes.extend([2]*int(mask.sum()))
    arrays=dict(obs=np.asarray(obs),next_obs=np.asarray(nxt),modes=np.asarray(modes))
    values=score.revalue_arrays(arrays,cfg)
    assert len(values)==15023 and np.isfinite(values).all()
    # Exercise the real FIFO/phase-finalization interface with synthetic test transitions only.
    writer_rows=[]
    class Writer:
        def writerow(self,row):writer_rows.append(dict(row))
    test_obs=arrays['obs'][0].copy();test_next=arrays['next_obs'][0].copy()
    test_obs[5:8]=[.04,-.03,0];test_next[5:8]=[.04,-.03,0]
    test_action=test_next[34:46].copy()
    buffer=PhaseFallBuffer(5,cfg)
    for index in range(2):
        row=dict(step=index+1,reward=1.,reward_fall=0.,motion_phase='walk',wz_cmd=0.,
                 _cycle_actual=[0.,0.,0.],_cycle_target=[0.,0.,0.],_cycle_dt=.04)
        buffer.append(test_obs,test_action,1.,test_next,index==1,{'mode':2},row,None)
    buffer.flush(agent.replay,Writer(),fell=True,complete_cycle=False)
    assert agent.replay.count==2 and len(writer_rows)==2
    assert agent.replay.online is agent.replay
    # Independent test instance validates wrapped FIFO retains only its latest data.
    small=OnlineFIFOReplay(3,1,0,55)
    for index in range(5):small.add(test_obs,test_action,float(index),test_next,False)
    assert small.count==3 and set(small.rewards[:,0])=={2.,3.,4.}
    small.sample(2);assert small.last_sample_counts['expert']==0
    # Forward and backward passes only: no Adam/target update or fitted checkpoint.
    batch=torch.as_tensor(arrays['obs'][:32],dtype=torch.float32).clone()
    batch[:,6]=torch.linspace(-.04,.04,len(batch))
    actions=agent.actor(batch)
    loss=agent.critic(torch.cat((batch,actions),dim=1)).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.actor.parameters() if p.grad is not None)
    assert all(torch.isfinite(p.grad).all() for p in agent.critic.parameters() if p.grad is not None)
    assert agent.updates==0 and not agent.actor_optimizer.state and not agent.critic_optimizer.state
    model=torch.load(OUT/'assets/actor.pth',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,model['actor'][k]) for k,v in agent.actor.state_dict().items())
    report=dict(optimizer_updates=0,simulation_started=False,training_started=False,critic_pretraining_started=False,
        command_steps_checked=3200,command_count=16,hold_control_steps=100,
        command_clock_survives_resets=True,reference_reward_max_error=max_error,
        reward_peak=peak,fall_backfill_matches_reference=True,
        actor_input=50,critic_input=61,online_replay_initial_count=0,expert_replay_count=0,
        critic_initialization='random',finite_forward_backward_without_updates=True,
        fifo_and_phase_flush_checked=True,old_walk_transitions_offline_scored=len(values),
        old_walk_score_mean=float(values.mean()),
        source_expert=str(source),source_expert_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        data_decision='Reuse original BC bytes; no expert/rollout transitions loaded into online Replay. Old Walk data is a scoring check only; no collection, relabel dataset or Critic pretraining needed.',
        coverage_note='Existing demonstration commands are not the16-command target coverage; no claimed side/back demonstration coverage.',
        total_env_steps=cfg['training']['total_env_steps'])
    (OUT/'offline_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
