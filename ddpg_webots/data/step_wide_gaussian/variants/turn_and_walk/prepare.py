"""Prepare derived data, zero-weight vy Actor expansion and offline checks only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
FOLDER=OUT.parents[1]
PROJECT=FOLDER.parents[1]
sys.path.insert(0,str(FOLDER.parent/'runtime'))
sys.path.insert(0,str(FOLDER.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.learning import load_policy, create_critic, create_algorithm
from basewalk.network_inputs import actor_indices, actor_input_dim, critic_input_dim
from basewalk.bc_rollout import compatible_actor_identities, identity
from basewalk.three_class_online import validate_online_experts
from basewalk.turn_and_walk import TurnAndWalkStages
from basewalk.gradual_turn import source_motion_stage, GradualTurnStages


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reward(path):
    spec=importlib.util.spec_from_file_location('scorer_'+sha(path),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    previous=json.loads((OUT/'previous_config.json').read_text(encoding='utf-8'))
    # Reuse the scoring helpers, replace only the complete motion formula.
    text=(FOLDER/'reward.py').read_text(encoding='utf-8')
    prefix=text[:text.index('def motion_reward(')]
    suffix=text[text.index('def observation_reward('):]
    motion='''def motion_reward(command, velocity, gyro_xy, gravity, mode, cfg, **unused):
    p = cfg['reward']['biped_tracking']
    values = np.r_[command, velocity, gyro_xy, gravity]
    if not np.isfinite(values).all() or mode not in (0, 1, 2):
        raise ValueError('Invalid tracking inputs')
    error = np.asarray(velocity) - np.asarray(command)
    x = p['turn_walk_xy_peak'] * math.exp(-float((error[0]/p['turn_walk_xy_width_m_s'])**2))
    y = p['turn_walk_xy_peak'] * math.exp(-float((error[1]/p['turn_walk_xy_width_m_s'])**2))
    yaw = p['turn_yaw_gaussian_weight'] * math.exp(-float((error[2]/p['turn_yaw_gaussian_width_rad_s'])**2))
    value = x + y + yaw
    return value, dict(velocity_tracking=value, stability=0., orientation=0.,
        orientation_raw=float(np.asarray(gravity)[:2] @ np.asarray(gravity)[:2]),
        turn_heading=0., tracking_xy_raw=x+y, tracking_yaw_raw=yaw,
        stability_raw=0., tracking_x_raw=x, tracking_y_raw=y,
        tracking_x=x, tracking_y=y, tracking_yaw=yaw,
        turn_total=value if mode==0 else 0., stand_total=value if mode==1 else 0.,
        walk_total=value if mode==2 else 0., turn_weighted=value if mode!=2 else 0.,
        walk_weighted=value if mode==2 else 0.)


'''
    (OUT/'reward.py').write_text(prefix+motion+suffix,encoding='utf-8')
    document=copy.deepcopy(previous)
    document.update(name='Turn and Walk - step wide Gaussian',status='prepared_no_training',
                    reward_file='variants/turn_and_walk/reward.py',
                    runs_root='variants/turn_and_walk/runs',reports_root='variants/turn_and_walk/reports')
    t=document['training_config']['training']
    t.pop('gradual_turn',None)
    t['online_turn_only']=False
    t['turn_and_walk']=dict(enabled=True,walk_attempts_per_level=4,
        vx_levels_m_s=[0,.02,.04,.06,.08,.06,.04,.02,0],
        vy_levels_m_s=[0,.01,.02,.03,.04,.03,.02,.01,0])
    active=document['training_config']
    active['command_vy_input']=True
    active['reward']['biped_tracking'].update(turn_walk_xy_peak=.625,turn_walk_xy_width_m_s=.10)
    write(FOLDER/'config.json',document)
    cfg=training_configuration(FOLDER/'config.json')
    old_cfg=copy.deepcopy(cfg);old_cfg.pop('command_vy_input')
    old_cfg['reward']=copy.deepcopy(previous['training_config']['reward'])
    source_actor=cfg['initialization']['bc_checkpoint']
    actor,actor_cfg,_=load_policy(source_actor)
    payload=torch.load(source_actor,map_location='cpu',weights_only=False)
    old_indices,new_indices=actor_indices(actor_cfg),actor_indices(cfg)
    assert new_indices==old_indices+(6,)
    weights=payload['actor']['network.0.weight']
    expanded=torch.zeros((weights.shape[0],len(new_indices)),dtype=weights.dtype)
    expanded[:,:len(old_indices)]=weights
    payload['actor']['network.0.weight']=expanded
    payload['config']=copy.deepcopy(cfg)
    payload['derivation']=dict(kind='zero_weight_command_vy_expansion',source=source_actor,
                              source_sha256=sha(source_actor),training_updates=0)
    for key in ('actor_target','critic','critic_target','actor_optimizer','critic_optimizer'):
        payload.pop(key,None)
    actor_path=OUT/'pretrain/actor.pth';actor_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,actor_path)
    new_actor,_,_=load_policy(actor_path)
    parent_source=copy.deepcopy(active['migration']['equivalent_actor_source'])
    active['migration']['equivalent_actor_source']=dict(path=previous['training_config']['initialization']['bc_checkpoint'],sha256=sha(source_actor))
    active['migration']['additional_equivalent_actor_sources']=[parent_source]
    for entry in (active['initialization'],active['three_class_online']):
        if entry is active['initialization']:
            entry.update(bc_checkpoint='variants/turn_and_walk/pretrain/actor.pth',bc_sha256=sha(actor_path))
        else:
            entry.update(actor='variants/turn_and_walk/pretrain/actor.pth',actor_sha256=sha(actor_path),critic_sha256='NOT_PRETRAINED')
    write(FOLDER/'config.json',document)
    cfg=training_config(training_configuration(FOLDER/'config.json'))
    identities=compatible_actor_identities(cfg)
    source_expert=cfg['three_class_online']['dataset'];source_rollout=cfg['bc_rollout']['source']['path']
    expert=torch.load(source_expert,map_location='cpu',weights_only=False)
    rollout=torch.load(source_rollout,map_location='cpu',weights_only=False)
    scorer=load_reward(OUT/'reward.py');old_scorer=load_reward(FOLDER/'variants/gradual_reversal/reward.py')
    maximum=0.;coverage=[];reward_changes=[]
    for episode in expert['episodes']:
        arrays=episode['arrays']
        old=np.asarray(episode['revalued_rewards'])
        new=scorer.revalue_arrays(arrays,cfg)
        reward_changes.extend((new-old).tolist())
        episode['revalued_rewards']=new
        obs=np.asarray(arrays['observations']);mode=np.asarray(arrays['modes'])
        coverage.append(dict(observations=obs,modes=mode))
    all_obs=np.concatenate([a['observations'] for a in coverage])
    all_modes=np.concatenate([a['modes'] for a in coverage])
    def compare_actions(obs):
        difference=0.
        with torch.no_grad():
            for start in range(0,len(obs),1024):
                batch=torch.as_tensor(obs[start:start+1024],dtype=torch.float32)
                difference=max(difference,float((actor(batch)-new_actor(batch)).abs().max()))
        return difference
    maximum=max(compare_actions(all_obs),compare_actions(np.asarray(rollout['arrays']['obs'])))
    assert maximum==0.
    expert['config']=copy.deepcopy(cfg)
    expert['derivation_turn_and_walk']=dict(source=source_expert,source_sha256=sha(source_expert),
        method='Unified x/y/yaw reward relabel only; physical states/actions/source actor unchanged')
    expert_path=PROJECT/'expert/step_wide_gaussian/turn_and_walk/expert.pth'
    expert_path.parent.mkdir(parents=True,exist_ok=True);torch.save(expert,expert_path)
    validate_online_experts(expert,cfg)
    arrays=rollout['arrays']
    before=old_scorer.revalue_arrays(arrays,old_cfg)
    after=scorer.revalue_arrays(arrays,cfg)
    old_stored=arrays['rewards'].clone()
    arrays['rewards']=old_stored+torch.as_tensor(after-before,dtype=old_stored.dtype).reshape(-1,1)
    residual_error=float(torch.max(torch.abs((arrays['rewards']-torch.as_tensor(after,dtype=old_stored.dtype).reshape(-1,1))-(old_stored-torch.as_tensor(before,dtype=old_stored.dtype).reshape(-1,1)))))
    rollout['derivation_turn_and_walk']=dict(source=source_rollout,source_sha256=sha(source_rollout),
        original_identity=rollout['identity'],original_config=rollout['config'],
        method='Reward labels/scoring identity derived; fall residual and source Actor/physical arrays retained')
    rollout['config']=copy.deepcopy(cfg)
    rollout['reward_source_sha256']=cfg['reward_source_sha256']
    rollout['identity']=identities[rollout['actor_sha256']]
    rollout_path=OUT/'pretrain/rollout.pth';torch.save(rollout,rollout_path)
    active['three_class_online'].update(dataset='../../expert/step_wide_gaussian/turn_and_walk/expert.pth',dataset_sha256=sha(expert_path))
    active['bc_rollout']['source']=dict(path='variants/turn_and_walk/pretrain/rollout.pth',sha256=sha(rollout_path))
    write(FOLDER/'config.json',document)
    cfg=training_config(training_configuration(FOLDER/'config.json'))
    constructor=dict(turn_only=False,count_fall_as_attempt=True,fixed_turn_sign=None,cycle_limit=20)
    stage=TurnAndWalkStages(cfg,**constructor)
    yaw=[.022,.044,.066,.088]+[.11]*20+[.088,.066,.044,.022,0,-.022,-.044,-.066,-.088]+[-.11]*20+[-.088,-.066,-.044,-.022,0,.022,.044,.066,.088]+[.066,.044,.022,0.]
    expected=[np.array([0,0,x],np.float32) for x in yaw]
    labels=['turn']*len(expected)
    for axis,levels in [(0,[0,.02,.04,.06,.08,.06,.04,.02,0]),(1,[0,.01,.02,.03,.04,.03,.02,.01,0])]:
        for value in levels:
            command=np.zeros(3,np.float32);command[axis]=value
            expected.extend([command]*4);labels.extend(['walk']*4)
    assert len(expected)==138
    assert np.abs(np.diff(np.r_[np.array(expected)[:,2],expected[0][2]])).max()<.022001
    for i in range(len(expected)*3):
        np.testing.assert_array_equal(stage.command(0,0),expected[i%len(expected)])
        assert stage.stage==labels[i%len(expected)]
        assert source_motion_stage(stage.command(0,0),cfg,stage.stage)==stage.stage
        prior=stage.state();stage.advance_phase(.1,.2);assert stage.state()==prior
        stage.advance_phase(.9,.95 if i%3==0 else .1,fell=i%3!=1)
        stage=TurnAndWalkStages(cfg,state=stage.state(),**constructor)
    for invalid in (dict(stage='walk',turn_walk_index=-1,cycles=0),
                    dict(stage='turn',turn_walk_index=0,cycles=1)):
        try:TurnAndWalkStages(cfg,state=invalid,**constructor)
        except ValueError:pass
        else:raise AssertionError('Invalid continuation accepted')
    critic=create_critic(cfg)
    sample=torch.as_tensor(all_obs[:32],dtype=torch.float32).clone();sample[:,6]=.04
    with torch.no_grad():
        new_actor(sample)
        critic(torch.cat([sample,torch.zeros((len(sample),12))],dim=1))
    assert actor_input_dim(cfg)==50 and critic_input_dim(cfg,0)==61
    for command in ([0,0,0],[0,.04,0],[.08,0,0],[0,0,-.11]):
        score=scorer.motion_reward(command,command,[0,0],[0,0,-1],2,cfg)[0]
        assert abs(score-3.25)<1e-12
    # Reject vy input on old Actor, preserve old constraints.
    try:actor(sample)
    except ValueError:pass
    else:raise AssertionError('Old zero-vy contract was lost')
    # A compatible BC initialization may be checked without a pretrained Critic.
    runtime=copy.deepcopy(cfg);runtime.pop('three_class_online')
    agent=create_algorithm(runtime);assert agent.updates==0
    values=dict(expert_transitions=len(all_obs),turn=int((all_modes==0).sum()),
        walk=int((all_modes==2).sum()),stand=int((all_modes==1).sum()),
        nonzero_vx_cmd=int((np.abs(all_obs[:,5])>1e-6).sum()),
        nonzero_vy_cmd=int((np.abs(all_obs[:,6])>1e-6).sum()))
    report=dict(actor_input=50,critic_input=61,shared_state_columns=list(actor_indices(cfg)),
        actor_action_difference=maximum,extra_vy_weights_zero=True,optimizer_updates=0,
        simulation_started=False,pretraining_started=False,attempts_per_loop=138,
        schedule_attempts_checked=414,zero_stage_matches_scheduler=True,
        fall_residual_max_error=residual_error,expert_coverage=values,
        expert=dict(source=source_expert,source_sha256=sha(source_expert),derived=str(expert_path),sha256=sha(expert_path)),
        rollout=dict(source=source_rollout,source_sha256=sha(source_rollout),derived=str(rollout_path),sha256=sha(rollout_path)),
        actor=dict(source=source_actor,source_sha256=sha(source_actor),derived=str(actor_path),sha256=sha(actor_path)),
        side_command_coverage='No prior nonzero vy_cmd demonstrations; reuse old physical data without fabricating lateral commands. Targeted coverage remains pending.',
        critic_status='NOT_PRETRAINED; new Reward and input require explicit authorized pretraining before online launch')
    write(OUT/'offline_check.json',report)
    write(OUT/'prepared_project_config.json',document)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
