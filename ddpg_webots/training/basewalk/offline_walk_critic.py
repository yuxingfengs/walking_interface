"""Offline Walk Critic fitting, independent of the online FIFO sampling policy."""
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import uuid
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(cfg):
    keys = ('reward', 'reward_source_sha256', 'environment', 'physical_gait',
            'actor_observation', 'command_vy_input', 'network_inputs', 'actor_type',
            'actor_hidden', 'critic_hidden', 'critic_observation', 'support_observation')
    values = {key: cfg.get(key) for key in keys}
    values['actor_sha256'] = cfg['initialization']['bc_sha256']
    values['gamma'] = cfg['training']['gamma']
    values['tau'] = cfg['training']['tau']
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def prepare(cfg):
    """Derive rewards from actual Walk transitions; never relabel source commands."""
    from .core import CONTRACT
    from .network_inputs import observation_dim
    from .reward_dispatch import revalue_arrays
    from .phase_fall import fall_penalty_at
    spec = cfg['offline_walk_critic']
    if type(spec['updates']) is not int or spec['updates'] <= 0:
        raise ValueError('Offline Critic updates must be a positive integer')
    source = spec['source']
    if digest(source['path']) != source['sha256']:
        raise ValueError('Offline Walk source hash mismatch')
    data = torch.load(source['path'], map_location='cpu', weights_only=False)
    if data.get('contract') != CONTRACT:
        raise ValueError('Offline Walk physical State contract mismatch')
    old = data['config']
    checks = ('physical_gait', 'support_observation', 'actor_observation',
              'command_vy_input', 'network_inputs', 'critic_observation')
    for key in checks:
        if old.get(key) != cfg.get(key):
            raise ValueError('Offline Walk source incompatible: ' + key)
    # Scheduling and task goals may differ; sensor, Action and dynamics settings may not.
    env_keys = ('yaw_rate_scale', 'action_scale', 'gait_frequency_hz',
                'joint_velocity_scale', 'motor_velocity_scale', 'joint_margin',
                'fall_height', 'fall_tilt_rad', 'floor_z')
    for key in env_keys:
        if old['environment'][key] != cfg['environment'][key]:
            raise ValueError('Offline Walk environment mismatch: ' + key)
    original = data['arrays']
    arrays = {k: np.asarray(v) for k, v in original.items()}
    n = len(arrays['obs']); width = observation_dim(cfg)
    for key, shape in (('obs',(n,width)), ('next_obs',(n,width)),
                       ('actions',(n,12)), ('terminal',(n,1))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid offline Walk array: ' + key)
    if not np.isin(arrays['terminal'], [0,1]).all():
        raise ValueError('Invalid terminal mask')
    ends = np.flatnonzero(arrays['cycle_ends'])
    selected=[]; rewards=[]; start=0; failed=0; segments=0
    for end in ends:
        stop=int(end)+1; indices=np.arange(start,stop)
        if np.all(arrays['modes'][indices] == 2):
            part={k:v[indices] for k,v in arrays.items()}
            score=revalue_arrays(part,cfg)
            fell=bool(part['terminal'][-1,0])
            if np.any(part['terminal'][:-1]):
                raise ValueError('Terminal inside recorded cycle')
            if fell:
                score += [fall_penalty_at((i+1)/len(indices),cfg['reward']['fall_penalty'])
                          for i in range(len(indices))]
                failed+=1
            selected.extend(indices.tolist());rewards.extend(score.tolist());segments+=1
        start=stop
    idx=np.asarray(selected,dtype=int)
    if len(idx)<cfg['training']['batch_size']:
        raise ValueError('Not enough compatible Walk transitions')
    output={k:torch.as_tensor(arrays[k][idx].copy(),dtype=torch.float32)
            for k in ('obs','actions','next_obs','terminal')}
    output['rewards']=torch.tensor(rewards,dtype=torch.float32).reshape(-1,1)
    if not all(torch.isfinite(v).all() for v in output.values()):
        raise ValueError('Nonfinite revalued Walk data')
    audit=dict(source=source, source_actor_sha256=data.get('actor_sha256'),
               current_actor_sha256=cfg['initialization']['bc_sha256'],
               source_kind=data.get('kind'), use='off_policy_physical_transitions',
               commands_relabelled=False, rows=len(idx),segments=segments,failed_segments=failed,
               source_rows=n, trailing_unfinalized_rows=n-start,
               commands_vx=np.unique(arrays['obs'][idx,5]).tolist(),
               reward_mean=float(np.mean(rewards)), signature=signature(cfg),
               optimizer_updates=0, decision='reuse_and_revalue_no_collection')
    path=Path(spec['dataset']);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(arrays=output,audit=audit),path)
    audit['dataset_sha256']=digest(path)
    path.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return output,audit


def initialize(agent,cfg):
    spec=cfg['offline_walk_critic']
    path=Path(spec['critic'])
    if not path.is_file() or spec.get('critic_sha256') == 'NOT_PRETRAINED':
        raise ValueError('Walk Critic is not pretrained. Use this experiment Pretrain Critic.cmd first.')
    if digest(path)!=spec['critic_sha256']:
        raise ValueError('Walk Critic checkpoint hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    source = spec['source']
    audited = saved['audit']['source']
    # A relocated rollout must still match the audited bytes and every metadata field.
    if audited['path'] != source['path'] and digest(source['path']) != audited['sha256']:
        raise ValueError('Relocated Walk Critic source hash mismatch')
    if (saved.get('kind')!='offline_walk_critic_v1' or saved['signature']!=signature(cfg)
            or saved['updates']!=spec['updates'] or not saved['actor_frozen_verified']
            or dict(audited, path=source['path']) != source):
        raise ValueError('Walk Critic Reward, source or model contract changed')
    if digest(spec['dataset'])!=saved['dataset_sha256']:
        raise ValueError('Walk Critic derived data hash mismatch')
    agent.critic.load_state_dict(saved['critic'])
    agent.critic_target.load_state_dict(saved['critic_target'])
    agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
    for group in agent.critic_optimizer.param_groups:group['lr']=cfg['training']['critic_lr']
    agent.pretrained_critic_updates=saved['updates']


def pretrain(cfg, *, check=False, no_register=False, result_file=None):
    from .learning import create_algorithm, Replay
    from .network_inputs import observation_dim
    from .expert_pipeline import atomic_save
    arrays,audit=prepare(cfg)
    if check:
        print(json.dumps(audit,indent=2));print('CHECK PASSED. No optimizer update or simulation.')
        return
    spec=cfg['offline_walk_critic'];runtime=copy.deepcopy(cfg);runtime.pop('offline_walk_critic')
    agent=create_algorithm(runtime)
    n=len(arrays['obs']);agent.replay=Replay(n,cfg['seed'],0,observation_dim(cfg))
    for key,value in arrays.items():getattr(agent.replay,key)[:]=value.numpy()
    agent.replay.count=n
    before=copy.deepcopy(agent.actor.state_dict())
    agent.actor.requires_grad_(False);agent.actor_target.requires_grad_(False)
    folder=Path(cfg['project_layout']['runs_root'])/('critic_pretrain_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    (folder/'source').mkdir()
    shutil.copyfile(cfg['reward_module_path'],folder/'source/experiment_reward.py')
    shutil.copyfile(__file__,folder/'source/offline_walk_critic.py')
    (folder/'effective_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with (folder/'critic_metrics.jsonl').open('w',encoding='utf-8') as stream:
        for step in range(spec['updates']):
            metrics=agent.update(critic_only=True,update_actor=False)
            if step==0 or (step+1)%100==0:
                stream.write(json.dumps(dict(step=step+1,**metrics))+'\n');stream.flush()
                print('Critic',step+1,'loss',metrics['critic_loss'],flush=True)
    if any(not torch.equal(value,agent.actor.state_dict()[key]) for key,value in before.items()):
        raise ValueError('Actor changed during Critic pretraining')
    path=folder/'critic.pth'
    payload=dict(kind='offline_walk_critic_v1',config=cfg,signature=signature(cfg),audit=audit,
                 dataset_sha256=digest(spec['dataset']),updates=agent.updates,actor_frozen_verified=True,
                 critic=agent.critic.state_dict(),critic_target=agent.critic_target.state_dict(),
                 critic_optimizer=agent.critic_optimizer.state_dict())
    atomic_save(payload,path)
    result=dict(critic=str(path.resolve()),critic_sha256=digest(path),updates=agent.updates)
    if result_file:Path(result_file).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not no_register:
        config_path=Path(cfg['project_layout']['config_path'])
        if digest(config_path)!=cfg['project_layout']['config_sha256']:
            raise ValueError('Config changed during pretraining; checkpoint preserved without registration')
        document=json.loads(config_path.read_text(encoding='utf-8-sig'))
        document['training_config']['offline_walk_critic'].update(result)
        temp=config_path.with_suffix('.tmp');temp.write_text(json.dumps(document,indent=2),encoding='utf-8');temp.replace(config_path)
    print('Critic saved:',path,'; online training was not started.')
