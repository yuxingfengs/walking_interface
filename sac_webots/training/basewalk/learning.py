"""Replay helpers, policy loading, and SAC construction."""
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
    if config.get('algorithm') != 'sac':
        raise ValueError('sac_turn is a standalone SAC-only runtime')
    from .sac import SAC
    return SAC(config)
