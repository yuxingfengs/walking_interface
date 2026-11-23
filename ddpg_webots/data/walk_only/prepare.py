"""Prepare an independent DDPG Walk-only configuration; never train."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[1]
SOURCE_ROOT=PROJECT/'sac_isaac_lab/experiments/training_trials/walk_only'
SOURCE=SOURCE_ROOT/'runs/train_20261121T065933Z_ad4d9a'
WIDE=OUT.parent/'step_wide_gaussian'
sys.path.insert(0,str(OUT.parent/'runtime'))
from project import training_configuration


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=SOURCE if (SOURCE/'effective_config.json').is_file() else sorted(SOURCE_ROOT.glob('runs/train_*/effective_config.json'))[-1].parent
    source_cfg=json.loads((source/'effective_config.json').read_text(encoding='utf-8'))
    assert source_cfg['sac']['actor_lr']==3e-6 and source_cfg['sac']['critic_lr']==3e-4
    assert source_cfg['training']['walk_commands']['hold_steps']==100
    assert source_cfg['reward']['biped_tracking']['walk_tracking_widths_m_s']==[.036,.036]
    base=training_configuration(WIDE/'config.json')
    source_actor=Path(base['initialization']['bc_checkpoint'])
    (OUT/'assets').mkdir(exist_ok=True)
    actor=OUT/'assets/actor.pth';shutil.copyfile(source_actor,actor)
    cfg=copy.deepcopy(base)
    for key in ('three_class_online','outcome_replay','bc_rollout','expert_pipeline','expert_library',
                'three_class_pipeline','bc_actor_reuse','bc_actor_reuse_physical','shared_preparation_module',
                'pretrain_directory','shared_expert_package','bc_actor_reuse','migration','project_layout',
                'expert_collection_stability','physical_expert_control'):
        cfg.pop(key,None)
    t=cfg['training']
    for key in ('checkpoint_start','turn_and_walk','gradual_turn','joint_update_every_gait_attempts',
                'joint_update_every_turn_attempts','actor_update_every_turn_attempts','stratified_replay',
                'warmup_expert_replay','success_expert_replay'):
        t.pop(key,None)
    t.update(total_env_steps=20000,actor_lr=3e-6,critic_lr=3e-4,tau=.005,gamma=.995,
        batch_size=256,replay_size=100000,learning_starts=1000,update_every=1,actor_update_every=1,
        critic_warmup_updates=0,critic_warmup_env_steps=0,online_fifo_only=True,
        online_fixed_cycle_limit=100,online_turn_only=False,online_fixed_turn_sign=None,
        online_angles_deg=[0],walk_commands=copy.deepcopy(source_cfg['training']['walk_commands']))
    cfg['reward']['biped_tracking'].update(copy.deepcopy(source_cfg['reward']['biped_tracking']))
    cfg['reward']['biped_tracking']['walk_yaw_use_raw_gyro']=True
    cfg['reward']['fall_penalty']=source_cfg['reward']['fall_penalty']
    cfg['initialization'].update(use_bc=True,bc_checkpoint='assets/actor.pth',bc_sha256=sha(actor),
        explicit_bc_artifact=True,reset_to_bc_on_failure=False)
    cfg['environment']['velocity_filter']=base['environment']['velocity_filter']
    # Keep Webots paths portable under ddpg; sources below remain provenance only.
    cfg['webots'].update(python='python',world_template='../runtime/worlds/direction_circle_10m.wbt')
    cfg.pop('reward_module_path',None)
    cfg.pop('reward_source_sha256',None)
    cfg.pop('reward_velocity_source_sha256',None)
    cfg.pop('experiment',None);cfg.pop('experiment_name',None)
    cfg['previous_training_run']=str(WIDE/'variants/turn_and_walk/forward_lateral/resume_30k_to_60k/runs/train_20261121T060052_2cbf68')
    cfg['replay_revision']='online_fifo_walk_only_v1'
    cfg['training_budget_note']='Fresh Walk-only DDPG,20k default; SAC-aligned objective/commands/step update; no expert Replay or Critic pretraining.'
    live_path_keys={'path','actor','critic','dataset','bc_checkpoint','source','source_config','state_migration_report','manifest','world_template'}
    def portable(value,key=''):
        if isinstance(value,dict):return {k:portable(v,k) for k,v in value.items()}
        if isinstance(value,list):return [portable(v,key) for v in value]
        if isinstance(value,str) and key in live_path_keys and Path(value).is_absolute():
            try:return os.path.relpath(value,OUT).replace('\\','/')
            except ValueError:return value
        return value
    cfg=portable(cfg)
    document=dict(experiment_id='walk_only',name='Walk Only',reward_file='reward.py',
        runs_root='runs',reports_root='reports',experiments_root='.',current_run=None,status='prepared_not_trained',
        report=dict(view='all',window_steps=80,stride=40,command_tolerance=.0001,cycle_group_size=2),
        origin=dict(reward_reference=str(source),actor_source=str(source_actor),
                    actor_sha256=sha(source_actor)),training_config=cfg)
    (OUT/'config.json').write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    reward=(source/'source/reward.py').read_text(encoding='utf-8')
    # The project's Webots Python is3.8; replace strict zip with explicit shape checking.
    reward=reward.replace('zip(obs, nxt, np.ravel(arrays[\'modes\']), velocities, strict=True)',
                          'zip(obs, nxt, np.ravel(arrays[\'modes\']), velocities)')
    reward=reward.replace('    velocities = filtered_reward_velocities(arrays, cfg)\n    return np.array',
        "    velocities = filtered_reward_velocities(arrays, cfg)\n    if not len(obs)==len(nxt)==len(arrays['modes'])==len(velocities):\n        raise ValueError('Unaligned Walk arrays')\n    return np.array")
    (OUT/'reward.py').write_text(reward,encoding='utf-8')
    for name,mode in [('Start Training','train'),('Check Training','check'),('Pretrain Critic','pretrain'),('Report','report')]:
        text='@echo off\n'
        text+=f'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\\runtime\\launch.ps1" -Config "%~dp0config.json" -Mode {mode}\n'
        text+='set "entryExit=%errorlevel%"\nif not "%entryExit%"=="0" pause\nexit /b %entryExit%\n'
        (OUT/(name+'.cmd')).write_text(text,encoding='ascii')
    record=dict(reference_run=str(source),originally_inspected_run=str(SOURCE),reference_config_sha256=sha(source/'effective_config.json'),
        reference_reward_sha256=sha(source/'source/reward.py'),local_reward_sha256=sha(OUT/'reward.py'),
        reference_config=source_cfg,source_actor=str(source_actor),actor_sha256=sha(actor),
        reference_status_at_inspection=json.loads((source/'status.json').read_text()),no_sac_changes=True,training_started=False)
    shutil.copyfile(source/'source/reward.py',OUT/'assets/sac_reference_reward.py')
    shutil.copyfile(source/'effective_config.json',OUT/'assets/sac_reference_config.json')
    shutil.copyfile(source/'source/basewalk/phase_fall.py',OUT/'assets/sac_reference_phase_fall.py')
    (OUT/'reference.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Prepared independent ddpg/walk_only; no optimizer updates, simulation or training.')


if __name__=='__main__':main()
