"""Frozen BC collection. Replay contains executed policy transitions, including failures."""
from pathlib import Path
import copy
import numpy as np
import torch
from .learning import Replay
from .core import CONTRACT, critic_extra_dim
from .network_inputs import observation_dim
from .expert_pipeline import atomic_save,digest


def identity(cfg):
    keys=('reward','stationary_reward','turn_stand_walk','environment','direction_training',
          'actor_type','actor_hidden','critic_hidden','support_observation','physical_gait',
          'contact_expert','velocity_objective','velocity_tracking','critic_observation','staged_navigation')
    result = dict({k:cfg[k] for k in keys},fixed_cycle_commands=cfg.get('fixed_cycle_commands'),
                  state_contract=CONTRACT)
    if cfg.get('reward',{}).get('biped_tracking',{}).get('velocity_source') == 'causal_mean':
        result['reward_history_sha256'] = cfg['reward_history_sha256']
    if cfg.get('network_inputs'):
        result['network_inputs'] = cfg['network_inputs']
    if cfg.get('actor_observation'):
        result['actor_observation'] = cfg['actor_observation']
    if cfg.get('reward_velocity_source_sha256'):
        result['reward_velocity_source_sha256']=cfg['reward_velocity_source_sha256']
    return result


def compatible_actor_identities(cfg):
    """Accept source assets only after proving a zero-column Actor expansion."""
    init = cfg['initialization']
    identities = {init['bc_sha256']: identity(cfg)}
    source = cfg.get('migration', {}).get('equivalent_actor_source')
    if source is None:
        return identities

    from .bc import compatible_contract
    from .network_inputs import actor_indices

    def load_verified(path, sha256):
        if digest(path) != sha256:
            raise ValueError('Equivalent Actor checkpoint hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('contract') != CONTRACT:
            raise ValueError('Equivalent Actor checkpoint interface mismatch')
        return payload

    previous = load_verified(source['path'], source['sha256'])
    current = load_verified(init['bc_checkpoint'], init['bc_sha256'])
    source_cfg = dict(cfg, actor_observation=previous['config'].get('actor_observation'))
    if (not compatible_contract(previous['config'], source_cfg)
            or not compatible_contract(current['config'], cfg)):
        raise ValueError('Equivalent Actor preprocessing mismatch')

    old_indices, new_indices = actor_indices(source_cfg), actor_indices(cfg)
    added = [i for i in new_indices if i not in old_indices]
    if not added or tuple(i for i in new_indices if i not in added) != old_indices:
        raise ValueError('Equivalent Actor must add inputs without replacing source inputs')
    old_weights, new_weights = previous['actor'], current['actor']
    first_layer = 'network.0.weight'
    if old_weights.keys() != new_weights.keys():
        raise ValueError('Equivalent Actor parameter keys changed')
    for name, weight in old_weights.items():
        expanded = new_weights[name]
        if name == first_layer:
            if expanded.shape != (weight.shape[0], len(new_indices)):
                raise ValueError('Equivalent Actor first-layer shape mismatch')
            if torch.count_nonzero(expanded[:, [new_indices.index(i) for i in added]]).item():
                raise ValueError('Equivalent Actor added weights must be zero')
            expanded = expanded[:, [new_indices.index(i) for i in old_indices]]
        if expanded.dtype != weight.dtype or not torch.equal(expanded, weight):
            raise ValueError(f'Equivalent Actor changed source parameter {name}')
    identities[source['sha256']] = identity(source_cfg)
    return identities


class RolloutReplay:
    def __init__(self,capacity,seed,critic_feature_dim=0,obs_dim=50):
        self.online=Replay(capacity,seed,critic_feature_dim,obs_dim)
        self.modes=np.empty(capacity,np.int8)
        self.experts=[]
        self.last_sample_counts={}

    @property
    def count(self):return self.online.count

    def ready(self,n,**kwargs):return self.count>=n

    def add_online(self,obs,action,reward,nxt,terminal,*,mode=None,**kwargs):
        if self.count>=self.online.capacity:raise ValueError('Collection would overwrite transitions')
        if mode not in (0,1,2):raise ValueError('Missing source phase')
        self.modes[self.count]=mode
        self.online.add(obs,action,reward,nxt,terminal,**kwargs)


def initialize_collection(agent,cfg):
    agent.replay=RolloutReplay(cfg['bc_rollout']['steps'],cfg['seed'],critic_extra_dim(cfg),observation_dim(cfg))
    agent.actor.eval().requires_grad_(False)
    agent.actor_target.eval().requires_grad_(False)
    agent.collection_actor=copy.deepcopy(agent.actor.state_dict())


def verify_frozen(agent,weights):
    for net in (agent.actor,agent.actor_target):
        if any(not torch.equal(v,net.state_dict()[k]) for k,v in weights.items()):
            raise ValueError('Frozen BC Actor changed')


def export_collection(agent,cfg,folder,step):
    verify_frozen(agent,agent.collection_actor)
    if agent.updates:raise ValueError('Collection performed optimizer updates')
    b=agent.replay.online;n=b.count
    if n!=step:raise ValueError('Missing finalized collection transitions')
    arrays={k:torch.from_numpy(getattr(b,k)[:n].copy()) for k in
            ('obs','actions','rewards','next_obs','terminal')}
    arrays['modes']=torch.from_numpy(agent.replay.modes[:n].copy())
    if critic_extra_dim(cfg):
        for key in ('critic_features', 'next_critic_features'):
            arrays[key] = torch.from_numpy(getattr(b,key)[:n].copy())
    atomic_save(dict(kind='frozen_bc_rollout_v31',contract=CONTRACT,config=cfg,
        identity=identity(cfg),actor_sha256=cfg['initialization']['bc_sha256'],
        actor_frozen_verified=True,optimizer_updates=0,steps=step,arrays=arrays,
        reward_semantics='per-step tracking with gait-cycle fall backfill; failures retained'),
        Path(folder)/'bc_rollout.pth')


def check_rollout_critic(saved, cfg, actor_identities):
    if saved.get('training_source')!='frozen_bc_rollout_v31':
        raise ValueError('V31 requires Critic trained on actual frozen-BC rollouts')
    # Exact original path -> verified local copy; checkpoint bytes/hashes stay unchanged.
    path=Path(cfg.get('migration', {}).get('asset_relocations', {}).get(saved['rollout_path'], saved['rollout_path']))
    if not path.is_file():
        raise FileNotFoundError('Critic source rollout is missing: '+str(path)+
            '. Do not substitute a different dataset. Run 03 Pretrain Critic.cmd with the current validated collection, then 04 Check DDPG.cmd.')
    if digest(path)!=saved['dataset_sha256']:raise ValueError('BC rollout hash mismatch')
    rollout_actor = saved.get('rollout_actor_sha256', saved['actor_sha256'])
    if saved['actor_sha256'] not in actor_identities or rollout_actor not in actor_identities:
        raise ValueError('BC rollout or Critic Actor identity mismatch')
    if saved['rollout_identity'] != actor_identities[rollout_actor]:
        raise ValueError('BC rollout task/Reward changed')
