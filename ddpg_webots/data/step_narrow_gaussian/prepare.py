"""Derive narrow EMA Reward labels from immutable State55 trajectories."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
WIDE=FOLDER.parent/'step_wide_gaussian'
sys.path[:0]=[str(FOLDER.parent/'runtime'),str(FOLDER.parent/'runtime/training')]
from project import training_configuration
from basewalk.bc_rollout import compatible_actor_identities


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    assert not (FOLDER/'data_reuse.json').exists(), 'Preparation already recorded'
    source=training_configuration(WIDE/'config.json')
    document=copy.deepcopy(json.loads((WIDE/'config.json').read_text()))
    (FOLDER/'pretrain').mkdir(exist_ok=True)
    c=copy.deepcopy(source)
    for k in ['project_layout','experiment','experiment_name','reward_module_path','launch_reference']:c.pop(k,None)
    c['webots']['world_template']='../runtime/worlds/direction_circle_10m.wbt'
    c['webots']['python']='python';c['webots']['port']=1241
    p=c['reward']['biped_tracking'];p.update(turn_yaw_gaussian_width_rad_s=.08,turn_yaw_use_raw_gyro=False)
    assert abs(-math.expm1(-.04/p['turn_yaw_lowpass_tau_s'])-.1)<1e-12
    assert 'cycle_yaw_scoring' not in p and 'high_score' not in c['outcome_replay']
    for label,src in [('actor',source['initialization']['bc_checkpoint']),
                       ('equivalent_actor',source['migration']['equivalent_actor_source']['path'])]:
        shutil.copy2(src,FOLDER/'pretrain'/f'{label}.pth')
    c['initialization']['bc_checkpoint']='pretrain/actor.pth'
    c['three_class_online'].update(actor='pretrain/actor.pth',dataset='../../expert/step_narrow_gaussian/expert.pth',
                                    dataset_sha256='PENDING_DERIVATION',critic='pretrain/NOT_PRETRAINED.pth',critic_sha256='NOT_PRETRAINED')
    c['migration']['equivalent_actor_source']['path']='pretrain/equivalent_actor.pth'
    c['migration']['asset_relocations']={str(FOLDER/'pretrain/rollout.pth'):'pretrain/rollout.pth'}
    c['bc_rollout']['source'].update(path='pretrain/rollout.pth',sha256='PENDING_DERIVATION')
    c['training'].update(total_env_steps=20000,save_best_reward_cycle=True)
    c['training'].pop('checkpoint_start',None)
    document.update(experiment_id='003',name='Step narrow gyro EMA Gaussian',training_config=c,
                    current_run=None,status='data_derivation',origin=dict(project='ddpg',experiment='step_wide_gaussian'))
    save(FOLDER/'config.json',document)
    cfg=training_configuration(FOLDER/'config.json')
    old_reward=module('_source_wide_reward',WIDE/'reward.py');new_reward=module('_new_narrow_reward',FOLDER/'reward.py')
    data=torch.load(source['three_class_online']['dataset'],map_location='cpu',weights_only=False)
    original_data_config=copy.deepcopy(data['config']);old_data_hash=sha(source['three_class_online']['dataset'])
    changed=0;max_history_error=0.;physical=hashlib.sha256()
    for ep in data['episodes']:
        a=ep['arrays'];before=new_reward.step_rewards(a,cfg);old=np.asarray(ep['revalued_rewards'])
        m=np.asarray(a['modes']).reshape(-1);assert np.allclose(before[m!=0],old[m!=0],rtol=1e-5,atol=1e-7)
        changed+=int(np.count_nonzero(np.abs(before-old)>1e-7));ep['revalued_rewards']=before
        o=np.asarray(a['observations']);n=np.asarray(a['next_observations']);rates=new_reward.filtered_reward_velocities(a,cfg)[:,2]
        mask=(m==0)&(n[:,52]>.5)&np.all(o[:,5:8]==n[:,5:8],axis=1)
        if mask.any():max_history_error=max(max_history_error,float(np.max(np.abs(rates[mask]-n[mask,51]))))
        for key in ['observations','next_observations','executed_actions','terminals','modes','cycle_ends','cycle_dt']:
            if key in a:physical.update(key.encode());physical.update(np.asarray(a[key]).tobytes())
    assert max_history_error<1e-6
    derivation=dict(source_path=source['three_class_online']['dataset'],source_sha256=old_data_hash,
                    generation_config=original_data_config,method='Reward relabel only; physical arrays and source policy retained')
    data['narrow_ema_derivation']=derivation
    data['config']=cfg;data['reward_source_sha256']=cfg['reward_source_sha256']
    expert=ROOT/'expert/step_narrow_gaussian/expert.pth';expert.parent.mkdir(exist_ok=True)
    torch.save(data,expert);expert_sha=sha(expert)
    rollout_path=Path(source['bc_rollout']['source']['path'])
    rollout=torch.load(rollout_path,map_location='cpu',weights_only=False);a=rollout['arrays']
    old_base=old_reward.step_rewards(a,source);new_base=new_reward.step_rewards(a,cfg)
    old_rewards=np.asarray(a['rewards']).reshape(-1).copy();residual=old_rewards.astype(float)-old_base
    revalued=old_rewards.astype(float)+new_base-old_base
    modes=np.asarray(a['modes']).reshape(-1);assert np.allclose(revalued[modes!=0],old_rewards[modes!=0],atol=1e-7)
    original_rollout_config=copy.deepcopy(rollout['config'])
    a['rewards']=torch.as_tensor(revalued[:,None],dtype=torch.float32)
    assert np.max(np.abs((np.asarray(a['rewards']).reshape(-1)-new_base)-residual))<2e-6
    rollout['narrow_ema_derivation']=dict(source_path=str(rollout_path),source_sha256=sha(rollout_path),
        generation_config=original_rollout_config,fall_residual_preserved=True)
    rollout['config']=cfg
    # Preserve the actual generator identity; the current BC is its verified
    # zero-column expansion, not a new physical rollout source.
    rollout['identity']=compatible_actor_identities(cfg)[rollout['actor_sha256']]
    rollout['reward_source_sha256']=cfg['reward_source_sha256']
    target=FOLDER/'pretrain/rollout.pth';torch.save(rollout,target);rollout_sha=sha(target)
    c['three_class_online']['dataset_sha256']=expert_sha;c['bc_rollout']['source']['sha256']=rollout_sha
    save(FOLDER/'config.json',document)
    # Physical identity check is independent of the new labels.
    catalog_path=ROOT/'expert/catalog.json';catalog=json.loads(catalog_path.read_text());physical_sha=physical.hexdigest()
    assert physical_sha==catalog['step_wide_gaussian']['physical_sha256']
    catalog['step_narrow_gaussian']=dict(path='step_narrow_gaussian/expert.pth',sha256=expert_sha,
        transitions=20165,physical_sha256=physical_sha,source_path=source['three_class_online']['dataset'],
        reward_labels='Per-step body gyro Z EMA alpha0.10; Gaussian peak2 width0.08; no cycle backfill')
    save(catalog_path,catalog)
    for title in ['Start Training.cmd','Pretrain Critic.cmd','Check Training.cmd','Report.cmd']:
        shutil.copy2(WIDE/title,FOLDER/title)
    audit=dict(collection_required=False,bc_retrain_required=False,critic_pretrain_required=True,
        expert_transitions=20165,rollout_transitions=20000,expert_changed_rewards=changed,
        ema_history_max_error=max_history_error,physical_sha256=physical_sha,
        actor_sha256=c['initialization']['bc_sha256'],source_expert=derivation['source_path'],
        source_expert_sha256=old_data_hash,derived_expert=str(expert),derived_expert_sha256=expert_sha,
        source_rollout=str(rollout_path),source_rollout_sha256=sha(rollout_path),derived_rollout=str(target),
        derived_rollout_sha256=rollout_sha,fall_residual_preserved=True,alpha=.1,width=.08,peak=2.,port=1241)
    save(FOLDER/'data_reuse.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if 'config' not in k},indent=2))


if __name__=='__main__':main()
