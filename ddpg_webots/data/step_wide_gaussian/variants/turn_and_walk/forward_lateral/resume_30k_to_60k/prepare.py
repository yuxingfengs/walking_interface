"""Reconstruct recorded replay and prepare a non-exact checkpoint branch; no updates."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

OUT=Path(__file__).resolve().parent
VARIANT=OUT.parent
WIDE=OUT.parents[3]
RUN=VARIANT/'runs/train_20261121T052059_f38fce'
sys.path.insert(0,str(WIDE.parent/'runtime'))
sys.path.insert(0,str(WIDE.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.learning import create_algorithm
from basewalk.turn_and_walk import TurnAndWalkStages


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg=training_config(training_configuration(WIDE/'config.json'))
    checkpoint=RUN/'checkpoints/step_00030000.pth'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['env_steps']==30000
    for key in ('reward','environment','actor_hidden','critic_hidden','actor_observation',
                'command_vy_input','outcome_replay','fixed_cycle_commands'):
        assert saved['config'][key]==cfg[key],key
    assert saved['config']['training']['turn_and_walk']==cfg['training']['turn_and_walk']
    agent=create_algorithm(cfg)
    assert agent.updates==0
    replay=agent.replay
    frame=pd.read_csv(RUN/'training.csv')
    assert len(frame)==30000 and frame.step.iloc[-1]==30000
    rows=frame.to_dict('records')
    stages=TurnAndWalkStages(cfg,turn_only=False,count_fall_as_attempt=True,fixed_turn_sign=None)
    previous_next=None;count=0;attempts=0
    with gzip.open(RUN/'transition_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for row,line in zip(rows,stream):
            trace=json.loads(line)
            assert trace['step']==row['step']
            obs=np.asarray(trace.get('source_observation',previous_next),dtype=np.float32)
            nxt=np.asarray(trace['next_observation'],dtype=np.float32)
            assert obs.shape==nxt.shape==(55,)
            np.testing.assert_allclose(obs[5:8],[row['vx_cmd'],row['vy_cmd'],row['wz_cmd']],atol=1e-7)
            np.testing.assert_allclose(stages.command(0,0),obs[5:8],atol=1e-7)
            action=nxt[34:46].copy()
            replay.add_online(obs,action,float(row['reward']),nxt,bool(row['bellman_terminal']),
                mode=0 if row['motion_phase']=='turn' else 2,cycle_outcome=row['replay_destination'])
            stages.advance_phase(row['gait_phase_before'],row['gait_phase_after'],fell=bool(row['fell']))
            if row['fell'] or row['gait_phase_after']<row['gait_phase_before']:attempts+=1
            previous_next=nxt;count+=1
        assert stream.readline()==''
    assert count==30000 and attempts==782
    assert replay.success.count==29586 and replay.failure.count==350 and replay.expert_count==16773
    assert replay.skipped_incomplete==64
    assert replay.ready(cfg['training']['batch_size'])
    replay_path=OUT/'reconstructed_replay.pth'
    torch.save(dict(source_checkpoint_sha256=sha(checkpoint),source_step=30000,
        reward_sha256=cfg['reward_source_sha256'],replay=replay,
        provenance=dict(csv_sha256=sha(RUN/'training.csv'),trace_sha256=sha(RUN/'transition_trace.jsonl.gz'),
            exact_sampling_rng_restored=False,original_source_run=str(RUN))),replay_path)
    for name in ('actor','critic','actor_target','critic_target','actor_optimizer','critic_optimizer'):
        getattr(agent,name).load_state_dict(saved[name])
    for name in ('actor','critic','actor_target','critic_target'):
        for key,value in getattr(agent,name).state_dict().items():
            assert torch.equal(value,saved[name][key])
    assert agent.actor_optimizer.param_groups[0]['lr']==cfg['training']['actor_lr']
    assert agent.critic_optimizer.param_groups[0]['lr']==cfg['training']['critic_lr']
    stage_state=stages.state()
    recovered=TurnAndWalkStages(cfg,state=stage_state,turn_only=False,count_fall_as_attempt=True,fixed_turn_sign=None)
    np.testing.assert_array_equal(stages.command(0,0),recovered.command(0,0))
    path=WIDE/'config.json';document=json.loads(path.read_text(encoding='utf-8'))
    document.update(name='Turn and Walk - forward lateral continued 30k to 60k',status='resume_prepared',
        runs_root='variants/turn_and_walk/forward_lateral/resume_30k_to_60k/runs',
        reports_root='variants/turn_and_walk/forward_lateral/resume_30k_to_60k/reports')
    document['training_config']['training'].update(total_env_steps=60000,checkpoint_start=dict(
        path='variants/turn_and_walk/forward_lateral/runs/train_20261121T052059_f38fce/checkpoints/step_00030000.pth',
        sha256=sha(checkpoint),step=30000,joint_attempts=attempts,fixed_cycle_state=stage_state,
        reconstructed_replay=dict(path='variants/turn_and_walk/forward_lateral/resume_30k_to_60k/reconstructed_replay.pth',sha256=sha(replay_path))))
    path.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    report=dict(source_run=str(RUN),source_checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        source_step=30000,target_step=60000,additional_env_steps=30000,
        saved_updates=saved['updates'],joint_attempts=attempts,fixed_cycle_state=stage_state,
        restored_command=stages.command(0,0).tolist(),
        replay_success=replay.success.count,replay_failure=replay.failure.count,replay_experts=replay.expert_count,
        skipped_incomplete=replay.skipped_incomplete,replay_path=str(replay_path),replay_sha256=sha(replay_path),
        networks_and_optimizers_verified=True,optimizer_updates_performed=0,
        exact_rng_resume=False,physical_world_reinitialized=True,
        decision='Reuse current command/reward/state and recorded Replay; resume network/target/Adam at30k without pretraining. This is a branch, not an exact simulation resume.')
    (OUT/'offline_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
