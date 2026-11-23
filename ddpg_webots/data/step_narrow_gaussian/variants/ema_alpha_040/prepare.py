"""Recompute only the causal yaw EMA State column and its Reward labels."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
EXPERIMENT=OUT.parents[1]
ROOT=EXPERIMENT.parents[1]
sys.path[:0]=[str(EXPERIMENT),str(ROOT/'ddpg/runtime'),str(ROOT/'ddpg/runtime/training')]
from project import training_configuration
from basewalk.bc_rollout import compatible_actor_identities
from basewalk.actors import create_actor
import reward as scoring


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def key(command):
    return (2 if abs(command[0])>1e-8 else 0 if abs(command[2])>1e-8 else 1,tuple(command.tolist()))


def states(arrays,alpha):
    obs=np.asarray(arrays.get('observations',arrays.get('obs')))
    nxt=np.asarray(arrays.get('next_observations',arrays.get('next_obs')))
    modes=np.asarray(arrays['modes']).reshape(-1)
    out=obs.copy();following=nxt.copy();value=None;filter_key=None
    for i in range(len(obs)):
        continuous=i>0 and not bool(np.asarray(arrays.get('terminals',arrays.get('terminal'))).reshape(-1)[i-1]) and np.allclose(obs[i],nxt[i-1],rtol=0,atol=1e-6)
        if not continuous:
            if modes[i]==0 and obs[i,52]>.5:
                raise ValueError('Partial Turn history cannot reconstruct new EMA')
            value=None;filter_key=None
        current_key=(int(modes[i]),tuple(obs[i,5:8].tolist()))
        if obs[i,52]<.5:value=None;filter_key=None
        out[i,51]=float(value[0]) if value is not None and filter_key==current_key else 0.
        if modes[i]==0:
            sample=np.asarray([nxt[i,50]],dtype=np.float32)
            if value is None or filter_key!=current_key:value=sample.copy()
            else:value+=alpha*(sample-value)
            filter_key=current_key
        else:value=None;filter_key=None
        following[i,51]=float(value[0]) if value is not None and filter_key==key(nxt[i,5:8]) else 0.
    return out,following


def replace_states(a,obs,nxt):
    for k,v in [('observations',obs),('next_observations',nxt)] if 'observations' in a else [('obs',obs),('next_obs',nxt)]:
        a[k]=torch.as_tensor(v,dtype=a[k].dtype) if torch.is_tensor(a[k]) else v.astype(a[k].dtype)


def main():
    assert not (OUT/'data_reuse.json').exists(), 'Already prepared'
    config_path=EXPERIMENT/'config.json';old=training_configuration(config_path)
    document=json.loads(config_path.read_text());save(OUT/'previous_config.json',document)
    if (EXPERIMENT/'execution.json').exists():save(OUT/'previous_execution.json',json.loads((EXPERIMENT/'execution.json').read_text()))
    c=document['training_config'];assert abs(-math.expm1(-.04/c['reward']['biped_tracking']['turn_yaw_lowpass_tau_s'])-.1)<1e-12
    c['reward']['biped_tracking']['turn_yaw_lowpass_tau_s']=-.04/math.log1p(-.4)
    c['three_class_online'].update(dataset='../../expert/step_narrow_gaussian/ema_alpha_040/expert.pth',dataset_sha256='PENDING_DERIVATION',
        critic='variants/ema_alpha_040/pretrain/NOT_PRETRAINED.pth',critic_sha256='NOT_PRETRAINED')
    rollout_target=OUT/'pretrain/rollout.pth'
    c['bc_rollout']['source'].update(path='variants/ema_alpha_040/pretrain/rollout.pth',sha256='PENDING_DERIVATION')
    c['migration'].setdefault('asset_relocations',{})[str(rollout_target)]='variants/ema_alpha_040/pretrain/rollout.pth'
    document['status']='ema_alpha_040_derivation';save(config_path,document)
    cfg=training_configuration(config_path)
    initial=torch.load(cfg['initialization']['bc_checkpoint'],map_location='cpu',weights_only=False)
    actor=create_actor(cfg);actor.load_state_dict(initial['actor']);actor.eval()
    expert_source=Path(old['three_class_online']['dataset']);data=torch.load(expert_source,map_location='cpu',weights_only=False)
    generator_config=copy.deepcopy(data['config']);ema_error=0.;action_error=0.;changed=0;raw=hashlib.sha256();keep=[i for i in range(55) if i!=51]
    for ep in data['episodes']:
        a=ep['arrays'];original_obs=np.asarray(a['observations']).copy();original_next=np.asarray(a['next_observations']).copy()
        old_obs,old_next=states(a,.1)
        ema_error=max(ema_error,float(np.max(np.abs(old_obs[:,51]-original_obs[:,51]))),float(np.max(np.abs(old_next[:,51]-original_next[:,51]))))
        new_obs,new_next=states(a,.4)
        assert np.array_equal(original_obs[:,keep],new_obs[:,keep]) and np.array_equal(original_next[:,keep],new_next[:,keep])
        with torch.no_grad():action_error=max(action_error,float((actor(torch.as_tensor(original_obs))-actor(torch.as_tensor(new_obs))).abs().max()))
        old_rewards=np.asarray(ep['revalued_rewards']).copy();replace_states(a,new_obs,new_next)
        new_rewards=scoring.revalue_arrays(a,cfg);m=np.asarray(a['modes']).reshape(-1)
        assert np.allclose(new_rewards[m!=0],old_rewards[m!=0],rtol=1e-5,atol=1e-7)
        ep['revalued_rewards']=new_rewards;changed+=int(np.count_nonzero(np.abs(new_rewards-old_rewards)>1e-7))
        raw.update(original_obs[:,keep].tobytes());raw.update(original_next[:,keep].tobytes())
    assert ema_error<1e-6 and action_error<1e-7
    data['ema_alpha_040_derivation']=dict(source_path=str(expert_source),source_sha256=sha(expert_source),
        generation_config=generator_config,derived_state_columns=[51],raw_physical_arrays_unchanged=True,
        source_alpha=.1,target_alpha=.4)
    data['config']=cfg;data['reward_source_sha256']=cfg['reward_source_sha256']
    expert_target=ROOT/'expert/step_narrow_gaussian/ema_alpha_040/expert.pth';expert_target.parent.mkdir(exist_ok=True)
    torch.save(data,expert_target);expert_sha=sha(expert_target)
    rollout_source=Path(old['bc_rollout']['source']['path']);rollout=torch.load(rollout_source,map_location='cpu',weights_only=False);a=rollout['arrays']
    old_base=scoring.revalue_arrays(a,old);stored=np.asarray(a['rewards']).reshape(-1).copy();residual=stored-old_base
    original_obs=np.asarray(a['obs']).copy();original_next=np.asarray(a['next_obs']).copy();o,n=states(a,.1)
    ema_error=max(ema_error,float(np.max(np.abs(o[:,51]-original_obs[:,51]))),float(np.max(np.abs(n[:,51]-original_next[:,51]))));assert ema_error<1e-6
    o,n=states(a,.4);assert np.array_equal(o[:,keep],original_obs[:,keep]) and np.array_equal(n[:,keep],original_next[:,keep])
    with torch.no_grad():
        for i in range(0,len(o),1024):action_error=max(action_error,float((actor(torch.as_tensor(original_obs[i:i+1024]))-actor(torch.as_tensor(o[i:i+1024]))).abs().max()))
    assert action_error<1e-7
    replace_states(a,o,n);new_base=scoring.revalue_arrays(a,cfg);a['rewards']=torch.as_tensor((stored+new_base-old_base)[:,None],dtype=torch.float32)
    assert np.max(np.abs(np.asarray(a['rewards']).reshape(-1)-new_base-residual))<2e-6
    rollout['ema_alpha_040_derivation']=dict(source_path=str(rollout_source),source_sha256=sha(rollout_source),
        generation_config=copy.deepcopy(rollout['config']),derived_state_columns=[51],fall_residual_preserved=True)
    rollout['config']=cfg;rollout['identity']=compatible_actor_identities(cfg)[rollout['actor_sha256']]
    rollout['reward_source_sha256']=cfg['reward_source_sha256'];torch.save(rollout,rollout_target)
    c['three_class_online']['dataset_sha256']=expert_sha;c['bc_rollout']['source']['sha256']=sha(rollout_target)
    save(config_path,document)
    catalog_path=ROOT/'expert/catalog.json';catalog=json.loads(catalog_path.read_text())
    catalog['step_narrow_gaussian_ema_040']=dict(path='step_narrow_gaussian/ema_alpha_040/expert.pth',sha256=expert_sha,
        transitions=20165,source_path=str(expert_source),source_sha256=sha(expert_source),derived_state_columns=[51],
        raw_physical_fields_unchanged=True,reward_labels='Per-step narrow Gaussian peak2 width0.08 with shared yaw EMA alpha0.40')
    save(catalog_path,catalog)
    audit=dict(collection_required=False,bc_retrain_required=False,critic_pretrain_required=True,
        expert_transitions=20165,rollout_transitions=20000,changed_expert_rewards=changed,
        old_alpha_reconstruction_max_error=ema_error,initial_actor_action_max_difference=action_error,
        raw_physical_fields_unchanged=True,derived_state_columns=[51],raw_physical_sha256=raw.hexdigest(),
        original_actor_sha256=cfg['initialization']['bc_sha256'],source_expert_sha256=sha(expert_source),
        derived_expert_sha256=expert_sha,source_rollout_sha256=sha(rollout_source),derived_rollout_sha256=sha(rollout_target),
        source_rollout_actor_sha256=rollout['actor_sha256'],fall_residual_preserved=True,alpha=.4,
        tau_s=c['reward']['biped_tracking']['turn_yaw_lowpass_tau_s'],width=.08,peak=2.)
    save(OUT/'data_reuse.json',audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
