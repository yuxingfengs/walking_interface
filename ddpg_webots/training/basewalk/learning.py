"""DDPG with executed 12D actions, terminal-aware targets and frozen target networks."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import (CONTRACT, DEFAULT_POSE, MINIMUM, MAXIMUM, OBS_DIM, validate,
                   critic_extra_dim, CRITIC_VELOCITY_FIELDS, CRITIC_REWARD_MEAN_FIELDS, FIELDS, vector)
from .network_inputs import observation_dim


def mlp(size, hidden, output):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(size, width), nn.ELU()))
        size = width
    layers.append(nn.Linear(size, output))
    return nn.Sequential(*layers)


def create_critic(config):
    from .network_inputs import compact, critic_input_dim, ProjectedCritic, SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY, actor_indices
    extra = critic_extra_dim(config)
    net = mlp(critic_input_dim(config,extra), config['critic_hidden'], 1)
    return ProjectedCritic(list(net.children()), extra, observation_dim(config),
                           actor_indices(config) if config.get('actor_observation') in (SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY) else None) if compact(config) else net


from .actors import create_actor


class Replay:
    def __init__(self, size, seed, critic_feature_dim=0, obs_dim=OBS_DIM):
        self.rng = np.random.default_rng(seed)
        self.capacity, self.count, self.cursor = size, 0, 0
        self.obs = np.empty((size, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty((size, 12), np.float32)
        self.rewards = np.empty((size, 1), np.float32)
        self.terminal = np.empty((size, 1), np.float32)
        self.critic_feature_dim = critic_feature_dim
        if critic_feature_dim:
            self.critic_features = np.empty((size, critic_feature_dim), np.float32)
            self.next_critic_features = np.empty_like(self.critic_features)

    def add(self, obs, action, reward, next_obs, terminated, expert=False, mode=None,
            critic_features=None, next_critic_features=None):
        i = self.cursor
        feature_dim = getattr(self, 'critic_feature_dim', 0)
        if feature_dim:
            current = vector(critic_features, feature_dim, 'Replay Critic features')
            following = vector(next_critic_features, feature_dim, 'Replay next Critic features')
            self.critic_features[i], self.next_critic_features[i] = current, following
        elif any(v is not None and np.asarray(v).size for v in (critic_features,next_critic_features)):
            raise ValueError('Critic features provided to a legacy Replay')
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminal[i] = next_obs, float(terminated)
        self.cursor = (i+1) % self.capacity
        self.count = min(self.count+1, self.capacity)

    def sample(self, n, use_expert=True, critic_only=False):
        arrays = (self.obs, self.actions, self.rewards, self.next_obs, self.terminal)
        if getattr(self, 'critic_feature_dim', 0):
            arrays += (self.critic_features, self.next_critic_features)
        if n == 0:
            return tuple(torch.from_numpy(v[:0].copy()) for v in arrays)
        idx = self.rng.integers(self.count, size=n)
        return tuple(torch.from_numpy(v[idx]) for v in arrays)

    def ready(self,n,critic_only=False):
        return self.count >= n


class OnlineFIFOReplay(Replay):
    """Online-only FIFO with the controller's diagnostic interface."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.last_sample_counts={}

    @property
    def online(self):return self

    def sample(self,n,**kwargs):
        result=super().sample(n,**kwargs)
        self.last_sample_counts=dict(online=n,expert=0)
        return result


class StratifiedReplay:
    """Persistent demonstrations plus balanced online turn/walk replay."""
    def __init__(self,size,seed,dataset,sha256,ratios):
        dataset=Path(dataset)
        if hashlib.sha256(dataset.read_bytes()).hexdigest()!=sha256:
            raise ValueError('Expert Replay hash mismatch')
        self.turn=Replay(size//2,seed)
        self.walk=Replay(size-size//2,seed+1)
        self.rng=np.random.default_rng(seed+2)
        with np.load(dataset,allow_pickle=False) as data:
            required=('observations','actions','rewards','next_observations','terminals','modes')
            if any(k not in data.files for k in required):
                raise ValueError('Expert Replay requires complete transitions and mode labels')
            self.expert={k:np.asarray(data[k]).copy() for k in required}
        n=len(self.expert['observations'])
        if (self.expert['observations'].shape!=(n,OBS_DIM) or self.expert['actions'].shape!=(n,12)
                or self.expert['next_observations'].shape!=(n,OBS_DIM)
                or self.expert['rewards'].shape!=(n,) or self.expert['terminals'].shape!=(n,)
                or self.expert['modes'].shape!=(n,) or not np.isfinite(np.concatenate((
                    self.expert['observations'].ravel(),self.expert['actions'].ravel(),
                    self.expert['next_observations'].ravel(),self.expert['rewards'].ravel()))).all()):
            raise ValueError('Invalid Expert Replay arrays')
        self.expert_indices={mode:np.flatnonzero(self.expert['modes']==mode) for mode in (0,1)}
        if any(len(v)==0 for v in self.expert_indices.values()):
            raise ValueError('Expert Replay must contain turn and walk transitions')
        self.ratios=ratios
        self.last_sample_counts={}

    @property
    def count(self):
        return self.turn.count+self.walk.count

    def add(self,obs,action,reward,next_obs,terminated,expert=False,mode=None):
        walking = bool(mode) if mode is not None else abs(float(obs[5]))>=1e-7
        (self.walk if walking else self.turn).add(
            obs,action,reward,next_obs,terminated)

    def ready(self,n,critic_only=False):
        return self.turn.count>0 and self.walk.count>0

    def _expert_sample(self,mode,n):
        idx=self.rng.choice(self.expert_indices[mode],size=n,replace=True)
        e=self.expert
        arrays=(e['observations'][idx],e['actions'][idx],e['rewards'][idx,None],
                e['next_observations'][idx],e['terminals'][idx,None])
        return tuple(torch.from_numpy(np.asarray(x,dtype=np.float32)) for x in arrays)

    def sample(self,n,use_expert=True,critic_only=False):
        ratios=self.ratios if use_expert else (.5,.5,0.,0.)
        raw=np.asarray(ratios,dtype=float)*n
        counts=np.floor(raw).astype(int)
        for i in np.argsort(-(raw-counts))[:n-int(counts.sum())]: counts[i]+=1
        ot,ow,et,ew=map(int,counts)
        groups=(self.turn.sample(ot),self.walk.sample(ow),
                self._expert_sample(0,et),self._expert_sample(1,ew))
        merged=tuple(torch.cat(parts,dim=0) for parts in zip(*groups))
        order=torch.from_numpy(self.rng.permutation(n))
        self.last_sample_counts=dict(online_turn=ot,online_walk=ow,expert_turn=et,expert_walk=ew)
        return tuple(x[order] for x in merged)


class WarmupExpertReplay:
    """Build immutable expert turn/walk buffers from the frozen BC Actor."""
    def __init__(self,size,seed,ratios):
        self.online_turn=Replay(size//2,seed)
        self.online_walk=Replay(size-size//2,seed+1)
        self.expert_turn=Replay(size//2,seed+2)
        self.expert_walk=Replay(size-size//2,seed+3)
        self.rng=np.random.default_rng(seed+4)
        self.ratios=ratios
        self.last_sample_counts={}

    @property
    def count(self):
        return sum(buffer.count for buffer in (self.online_turn,self.online_walk,
                                                self.expert_turn,self.expert_walk))

    def add(self,obs,action,reward,next_obs,terminated,expert=False,mode=None):
        walking=bool(mode) if mode is not None else abs(float(obs[5]))>=1e-7
        if expert:
            buffer=self.expert_walk if walking else self.expert_turn
        else:
            buffer=self.online_walk if walking else self.online_turn
        buffer.add(obs,action,reward,next_obs,terminated)

    def ready(self,n,critic_only=False):
        if critic_only:
            return self.expert_turn.count>0 and self.expert_walk.count>0
        return (self.online_turn.count>0 and self.online_walk.count>0
                and self.expert_turn.count>0 and self.expert_walk.count>0)

    def sample(self,n,use_expert=True,critic_only=False):
        ratios=(0.,0.,.5,.5) if critic_only else self.ratios
        raw=np.asarray(ratios,dtype=float)*n
        counts=np.floor(raw).astype(int)
        for i in np.argsort(-(raw-counts))[:n-int(counts.sum())]: counts[i]+=1
        ot,ow,et,ew=map(int,counts)
        groups=(self.online_turn.sample(ot),self.online_walk.sample(ow),
                self.expert_turn.sample(et),self.expert_walk.sample(ew))
        merged=tuple(torch.cat(parts,dim=0) for parts in zip(*groups))
        order=torch.from_numpy(self.rng.permutation(n))
        self.last_sample_counts=dict(online_turn=ot,online_walk=ow,
                                     expert_turn=et,expert_walk=ew)
        return tuple(x[order] for x in merged)


class TurnStandWarmupReplay:
    """Turn-only Replay with a small, explicit aligned-Stand stratum."""
    def __init__(self,size,seed,ratios,warmup_ratios=(.9,.1),critic_feature_dim=0):
        capacities=[size//4+int(i<size%4) for i in range(4)]
        self.online_turn=Replay(capacities[0],seed,critic_feature_dim)
        self.online_stand=Replay(capacities[1],seed+1,critic_feature_dim)
        self.expert_turn=Replay(capacities[2],seed+2,critic_feature_dim)
        self.expert_stand=Replay(capacities[3],seed+3,critic_feature_dim)
        self.rng=np.random.default_rng(seed+4)
        self.ratios=tuple(ratios)
        self.warmup_ratios=tuple(warmup_ratios)
        self.last_sample_counts={}

    @property
    def count(self):
        return sum(buffer.count for buffer in (
            self.online_turn,self.online_stand,self.expert_turn,self.expert_stand))

    def add(self,obs,action,reward,next_obs,terminated,expert=False,mode=None,
            critic_features=None,next_critic_features=None):
        if type(mode) is not int or mode not in (0,1):
            raise ValueError('Turn/Stand Replay requires mode 0 (turn) or 1 (stand)')
        if expert:
            buffer=self.expert_stand if mode else self.expert_turn
        else:
            buffer=self.online_stand if mode else self.online_turn
        buffer.add(obs,action,reward,next_obs,terminated,
                   critic_features=critic_features,next_critic_features=next_critic_features)

    def ready(self,n,critic_only=False):
        required=((self.expert_turn,self.expert_stand) if critic_only else
                  (self.online_turn,self.online_stand,self.expert_turn,self.expert_stand))
        return all(buffer.count>0 for buffer in required)

    @staticmethod
    def _counts(ratios,n):
        raw=np.asarray(ratios,dtype=float)*n
        counts=np.floor(raw).astype(int)
        for i in np.argsort(-(raw-counts),kind='stable')[:n-int(counts.sum())]:
            counts[i]+=1
        return counts

    def sample(self,n,use_expert=True,critic_only=False):
        ratios=((0.,0.,*self.warmup_ratios) if critic_only else
                self.ratios if use_expert else (.9,.1,0.,0.))
        counts=self._counts(ratios,n)
        buffers=(self.online_turn,self.online_stand,self.expert_turn,self.expert_stand)
        groups=tuple(buffer.sample(int(count)) for buffer,count in zip(buffers,counts))
        merged=tuple(torch.cat(parts,dim=0) for parts in zip(*groups))
        order=torch.from_numpy(self.rng.permutation(n))
        names=('online_turn','online_stand','expert_turn','expert_stand')
        self.last_sample_counts={name:int(count) for name,count in zip(names,counts)}
        return tuple(x[order] for x in merged)


class SuccessfulExpertReplay:
    """One mixed FIFO plus protected Turn/Stand data from successful frozen-BC episodes."""
    def __init__(self, size, seed, ratios, capacity_ratios, critic_feature_dim=0):
        capacities = TurnStandWarmupReplay._counts(capacity_ratios, size)
        self.online = Replay(int(capacities[0]), seed, critic_feature_dim)
        self.expert_turn = Replay(int(capacities[1]), seed+1, critic_feature_dim)
        self.expert_stand = Replay(int(capacities[2]), seed+2, critic_feature_dim)
        self.rng = np.random.default_rng(seed+3)
        self.ratios = tuple(ratios)
        self.last_sample_counts = {}
        self.successes_by_angle = {}
        self._pending = []
        self._episode_is_expert = True

    @property
    def count(self):
        return self.online.count + self.expert_turn.count + self.expert_stand.count

    def expert_ready(self):
        return self.expert_turn.count > 0 and self.expert_stand.count > 0

    def add(self, obs, action, reward, next_obs, terminated, expert=False, mode=None,
            critic_features=None, next_critic_features=None):
        if type(mode) is not int or mode not in (0, 1):
            raise ValueError('Successful Expert Replay requires source Turn/Stand labels')
        self.online.add(obs, action, reward, next_obs, terminated,
                        critic_features=critic_features, next_critic_features=next_critic_features)
        if getattr(self, "fixed_experts", False):
            return
        self._episode_is_expert = self._episode_is_expert and bool(expert)
        if not self._episode_is_expert:
            self._pending.clear()
        else:
            # Keep the whole candidate trajectory until its outcome is known.
            # Copy values because simulator adapters may reuse their arrays.
            self._pending.append((mode, copy.deepcopy((obs, action, reward, next_obs, terminated,
                                                       critic_features, next_critic_features))))

    def finish_episode(self, success, angle):
        accepted = bool(success and self._episode_is_expert and self._pending)
        added = [0, 0]
        if accepted:
            for mode, transition in self._pending:
                obs, action, reward, nxt, terminal, features, next_features = transition
                target = self.expert_stand if mode else self.expert_turn
                target.add(obs, action, reward, nxt, terminal,
                           critic_features=features, next_critic_features=next_features)
                added[mode] += 1
            label = str(angle)
            self.successes_by_angle[label] = self.successes_by_angle.get(label, 0) + 1
        result = dict(admitted=accepted, expert_episode=self._episode_is_expert,
                      added_turn=added[0], added_stand=added[1],
                      online_count=self.online.count, expert_turn_count=self.expert_turn.count,
                      expert_stand_count=self.expert_stand.count,
                      successes_by_angle=dict(self.successes_by_angle))
        self._pending.clear()
        self._episode_is_expert = True
        return result

    def ready(self, n, critic_only=False):
        return self.online.count >= n and self.expert_ready()

    def sample(self, n, use_expert=True, critic_only=False):
        if not use_expert:
            raise ValueError('Successful Expert Replay always preserves expert sampling quotas')
        if not self.ready(n, critic_only):
            raise ValueError('Successful Expert Replay is not ready; no quota substitution allowed')
        counts = TurnStandWarmupReplay._counts(self.ratios, n)
        groups = tuple(buffer.sample(int(count)) for buffer, count in zip(
            (self.online, self.expert_turn, self.expert_stand), counts))
        merged = tuple(torch.cat(parts, dim=0) for parts in zip(*groups))
        order = torch.from_numpy(self.rng.permutation(n))
        self.last_sample_counts = dict(zip(('online_mixed', 'expert_turn', 'expert_stand'),
                                          (int(count) for count in counts)))
        return tuple(value[order] for value in merged)


class ThreeBucketWarmupReplay:
    """Frozen-Actor and online Replay stratified by low/mixed/high turn weight."""
    labels = ('low_turn', 'mixed', 'high_turn')

    def __init__(self, size, seed, ratios, boundaries):
        capacities = [size//3 + int(i < size % 3) for i in range(3)]
        self.online = [Replay(capacity, seed+i) for i, capacity in enumerate(capacities)]
        self.expert = [Replay(capacity, seed+3+i) for i, capacity in enumerate(capacities)]
        self.rng = np.random.default_rng(seed+6)
        self.ratios = tuple(ratios)
        self.boundaries = tuple(boundaries)
        self.last_sample_counts = {}

    @property
    def count(self):
        return sum(buffer.count for buffer in (*self.online, *self.expert))

    def add(self, obs, action, reward, next_obs, terminated, expert=False, mode=None):
        if type(mode) is not int or not 0 <= mode < 3:
            raise ValueError('Three-bucket Replay requires mode 0, 1 or 2')
        buffers = self.expert if expert else self.online
        buffers[mode].add(obs, action, reward, next_obs, terminated)

    def ready(self, n, critic_only=False):
        required = self.expert if critic_only else (*self.online, *self.expert)
        return all(buffer.count > 0 for buffer in required)

    @staticmethod
    def _counts(ratios, n):
        raw = np.asarray(ratios, dtype=float)*n
        counts = np.floor(raw).astype(int)
        for i in np.argsort(-(raw-counts), kind='stable')[:n-int(counts.sum())]:
            counts[i] += 1
        return counts

    def sample(self, n, use_expert=True, critic_only=False):
        if critic_only:
            ratios = (0., 0., 0., 1/3, 1/3, 1/3)
        elif use_expert:
            ratios = self.ratios
        else:
            ratios = (1/3, 1/3, 1/3, 0., 0., 0.)
        counts = self._counts(ratios, n)
        buffers = (*self.online, *self.expert)
        groups = tuple(buffer.sample(int(count)) for buffer, count in zip(buffers, counts))
        merged = tuple(torch.cat(parts, dim=0) for parts in zip(*groups))
        order = torch.from_numpy(self.rng.permutation(n))
        names = tuple('online_'+label for label in self.labels) + tuple(
            'expert_'+label for label in self.labels)
        self.last_sample_counts = {name:int(count) for name, count in zip(names, counts)}
        return tuple(x[order] for x in merged)


def bellman_target(rewards, terminal, next_q, gamma):
    # Episode time limits/bootstrap: only a physical terminal masks future value.
    return rewards + gamma*(1.-terminal)*next_q


class DDPG:
    def __init__(self, config):
        self.config = validate(config)
        torch.manual_seed(config['seed'])
        torch.set_num_threads(1)
        self.actor = create_actor(config)
        self.critic = create_critic(config)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        t = config['training']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=t['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=t['critic_lr'])
        stratified=t.get('stratified_replay')
        warmup_expert=t.get('warmup_expert_replay')
        successful_expert=t.get('success_expert_replay')
        self.replay = None if config.get('expert_pipeline') else (SuccessfulExpertReplay(t['replay_size'],config['seed']+1,
                          successful_expert['ratios'],successful_expert['capacity_ratios'],
                          critic_feature_dim=critic_extra_dim(config)) if successful_expert else
                      ((TurnStandWarmupReplay(t['replay_size'],config['seed']+1,
                          warmup_expert['ratios'], warmup_expert.get('warmup_ratios', (.9,.1)),
                          critic_feature_dim=critic_extra_dim(config))
                        if warmup_expert.get('mode')=='turn_stand' else
                        ThreeBucketWarmupReplay(t['replay_size'], config['seed']+1,
                          warmup_expert['ratios'], warmup_expert['turn_weight_boundaries'])
                        if warmup_expert.get('turn_weight_boundaries') else
                        WarmupExpertReplay(t['replay_size'],config['seed']+1,
                          warmup_expert['ratios'])) if warmup_expert else
                       StratifiedReplay(t['replay_size'],config['seed']+1,
                          stratified['expert_dataset'],stratified['expert_sha256'],stratified['ratios'])
                          if stratified else Replay(t['replay_size'], config['seed']+1, critic_extra_dim(config),
                                                    observation_dim(config))))
        if t.get('online_fifo_only'):
            if any(config.get(k) for k in ('expert_pipeline','three_class_online','outcome_replay')) or any((stratified,warmup_expert,successful_expert)):
                raise ValueError('Online FIFO cannot mix expert or outcome Replay')
            self.replay=OnlineFIFOReplay(t['replay_size'],config['seed']+1,critic_extra_dim(config),observation_dim(config))
        self.rng = np.random.default_rng(config['seed']+2)
        self.updates = 0
        self.warmup_completed = 0
        self.bc_state = None
        self.bc_reference = None
        init = config.get('initialization', {})
        actor_checkpoint = init.get('actor_checkpoint')
        if actor_checkpoint:
            path = Path(actor_checkpoint)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != init.get('actor_checkpoint_sha256'):
                raise ValueError('Actor warm-start checkpoint hash mismatch')
            policy, saved, source_steps = load_policy(path)
            from .bc import compatible_contract
            if not compatible_contract(saved,config,bool(init.get('allow_yaw_source_change',False))):
                raise ValueError('Actor warm-start checkpoint is incompatible')
            self.actor.load_state_dict(policy.state_dict())
            self.actor_target.load_state_dict(policy.state_dict())
            self.initial_actor_source = dict(checkpoint=str(path), sha256=actual,
                                             source_steps=source_steps)
        if init.get('bc_checkpoint'):
            if actor_checkpoint:
                raise ValueError('Choose BC initialization or Actor warm start, not both')
            path = Path(init['bc_checkpoint'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != init['bc_sha256']:
                raise ValueError('BC checkpoint hash mismatch')
            policy, saved, _ = load_policy(path)
            from .bc import compatible_contract
            if not compatible_contract(saved,config,bool(init.get('allow_yaw_source_change',False))):
                raise ValueError('BC preprocessing/command/Actor configuration mismatch')
            self.bc_state = copy.deepcopy(policy.state_dict())
            self.bc_reference = copy.deepcopy(policy).eval().requires_grad_(False)
            self.restore_bc()
        from .expert_pipeline import initialize
        initialize(self, config)
        if config.get('bc_rollout', {}).get('collection_mode'):
            from .bc_rollout import initialize_collection
            initialize_collection(self, config)
        elif config.get('three_class_online'):
            from .three_class_online import initialize as initialize_three_class
            initialize_three_class(self, config)
        if config.get('offline_walk_critic'):
            from .offline_walk_critic import initialize
            initialize(self, config)

    def restore_bc(self):
        if self.bc_state is None:
            raise ValueError('No BC initialization available')
        self.actor.load_state_dict(self.bc_state)
        self.actor_target.load_state_dict(self.bc_state)
        lr = self.actor_optimizer.param_groups[0]['lr']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.warmup_completed = 0
        # Critic, its target/optimizer and the entire Replay Buffer are retained.

    def schedule(self, step):
        t = self.config['training']
        spec = t.get('success_expert_replay', {})
        if 'late_ratios' in spec and self.config.get('expert_pipeline', {}).get('mode') == 'train':
            self.replay.ratios = tuple(spec['late_ratios'] if step > spec['switch_step'] else spec['ratios'])
        if step > t.get('switch_steps', float('inf')):
            for group in self.actor_optimizer.param_groups:
                group['lr'] = t['actor_lr_after_switch']
            self.noise_std = t['exploration_std_after_switch']
        else:
            self.noise_std = t['exploration_std']

    def act(self, obs, explore=False):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(obs).float().unsqueeze(0))[0]
            if explore:
                noise = self.rng.normal(0., getattr(self, 'noise_std', self.config['training']['exploration_std']), 12)
                a = self.actor.project(a + torch.tensor(noise, dtype=torch.float32))
            return a.numpy()

    def policy_action(self, obs, action):
        if self.config.get('landing_alignment'):
            from .landing_projection import project_landing_actions
            return project_landing_actions(obs, action, self.config)
        return action

    def update(self, critic_only=False, update_actor=True):
        t = self.config['training']
        use_expert = not t.get('expert_replay_actor_freeze_only', False) or critic_only
        batch = self.replay.sample(
            t['batch_size'], use_expert=use_expert, critic_only=critic_only)
        obs, a, r, nxt, terminal = batch[:5]
        critic_obs, critic_nxt = obs, nxt
        if critic_extra_dim(self.config):
            features, next_features = batch[5:]
            critic_obs = torch.cat((obs, features), -1)
            critic_nxt = torch.cat((nxt, next_features), -1)
        with torch.no_grad():
            target_q = self.critic_target(torch.cat((critic_nxt, self.policy_action(nxt, self.actor_target(nxt))), -1))
            target = bellman_target(r, terminal, target_q, t['gamma'])
        predicted_q = self.critic(torch.cat((critic_obs, a), -1))
        critic_loss = nn.functional.mse_loss(predicted_q, target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('Nonfinite Critic loss; update stopped before optimizer step')
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1., error_if_nonfinite=True)
        self.critic_optimizer.step()
        actor_loss = None
        actor_rl_loss = None
        actor_bc_loss = None
        actor_grad = None
        if not critic_only and update_actor:
            self.critic.requires_grad_(False)
            actor_actions = self.policy_action(obs, self.actor(obs))
            actor_rl_loss = -self.critic(torch.cat((critic_obs, actor_actions), -1)).mean()
            coefficient = t.get('actor_bc_regularization', 0.)
            if coefficient:
                if self.bc_reference is None:
                    raise ValueError('Actor BC regularization requires BC initialization')
                with torch.no_grad():
                    reference_actions = self.policy_action(obs, self.bc_reference(obs))
                actor_bc_loss = nn.functional.mse_loss(actor_actions, reference_actions)
            else:
                actor_bc_loss = torch.zeros((), dtype=actor_rl_loss.dtype)
            actor_loss = actor_rl_loss + coefficient*actor_bc_loss
            if not torch.isfinite(actor_loss):
                raise FloatingPointError('Nonfinite Actor loss; Actor update stopped')
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1., error_if_nonfinite=True)
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)
        elif critic_only:
            self.warmup_completed += 1
        with torch.no_grad():
            for source, dest in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                for p, q in zip(source.parameters(), dest.parameters()):
                    q.lerp_(p, t['tau'])
        self.updates += 1
        sampled=getattr(self.replay,'last_sample_counts',{})
        return dict(actor_loss=float(actor_loss.detach()) if actor_loss is not None else None,
                    actor_rl_loss=float(actor_rl_loss.detach()) if actor_rl_loss is not None else None,
                    actor_bc_loss=float(actor_bc_loss.detach()) if actor_bc_loss is not None else None,
                    critic_loss=float(critic_loss.detach()), updates=self.updates,
                    q_mean=float(predicted_q.detach().mean()), target_q_mean=float(target.mean()),
                    td_abs_mean=float((predicted_q.detach()-target).abs().mean()),
                    critic_grad_norm=float(critic_grad),
                    actor_grad_norm=float(actor_grad) if actor_grad is not None else None,
                    replay_online_mixed=sampled.get('online_mixed'),
                    replay_online_turn=sampled.get('online_turn'),
                    replay_expert_turn=sampled.get('expert_turn'),
                    replay_expert_stand=sampled.get('expert_stand'),
                    replay_expert_walk=sampled.get('expert_walk'),
                    replay_expert_turn_stand_walk=sampled.get('expert_turn_stand_walk'))

    def save(self, path, steps):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(contract=CONTRACT, config=self.config, env_steps=steps, updates=self.updates,
            pretrained_critic_updates=getattr(self, 'pretrained_critic_updates', 0),
            reusable_expert_source=getattr(self, 'reusable_expert_source', None),
            bc_source=getattr(self, 'bc_source', None),
            initial_actor_source=getattr(self, 'initial_actor_source', None),
            critic_warmup_completed=self.warmup_completed,
            actor=self.actor.state_dict(), critic=self.critic.state_dict(),
            actor_target=self.actor_target.state_dict(), critic_target=self.critic_target.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            checkpoint_kind='evaluation_only_no_replay_resume')
        payload['critic_contract'] = dict(
            observation_fields=list(FIELDS),
            extra_feature_dim=critic_extra_dim(self.config),
            action_dim=12, input_dim=OBS_DIM+critic_extra_dim(self.config)+12,
            velocity_source='causal_position_difference_body_frame_unfiltered_m_s'
                if critic_extra_dim(self.config) else None)
        if self.config.get('critic_observation') == 'actor_plus_reward_velocity_mean':
            payload['critic_contract'].update(extra_fields=list(CRITIC_REWARD_MEAN_FIELDS),
                velocity_source='past_completed_steps_command_segment_mean',
                window_samples=self.config['velocity_tracking']['window_samples'])
        from .network_inputs import compact, OBS_KEEP, ACTION_KEEP, critic_input_dim, actor_indices, SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY
        payload['actor_contract'] = dict(input_dim=len(actor_indices(self.config)),
            observation_indices=list(actor_indices(self.config)),
            actor_observation=self.config.get('actor_observation'))
        if compact(self.config):
            payload['critic_contract'].update(input_dim=critic_input_dim(self.config,critic_extra_dim(self.config)),
                physical_packed_dim=observation_dim(self.config)+critic_extra_dim(self.config)+12,
                network_inputs=self.config['network_inputs'],observation_indices=list(actor_indices(self.config) if self.config.get('actor_observation') in (SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY) else OBS_KEEP),
                action_indices=list(ACTION_KEEP))
        # Never claim exact training resume without Replay and environment/RNG state.
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)


def load_policy(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if data.get('contract') != CONTRACT:
        raise ValueError('Checkpoint incompatible: V34 requires a 50D model; migrate 50D assets explicitly')
    cfg = validate(data['config'])
    if data.get('kind') == 'sac_turn_checkpoint_v1':
        from .sac import SquashedGaussianActor
        actor = SquashedGaussianActor(cfg, data['sac_spec'])
        actor.load_state_dict(data['actor'], strict=True)
        actor.eval()
        return actor, cfg, data['env_steps']
    actor = create_actor(cfg)
    state = copy.deepcopy(data['actor'])
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor, cfg, data['env_steps']


def create_algorithm(config):
    validate(config)
    if config.get('algorithm', 'ddpg') == 'sac':
        from .sac import SAC
        return SAC(config)
    return DDPG(config)
