"""FIFO replay classified by completed gait-cycle outcome, not data source."""
import numpy as np
import torch
from .learning import Replay


class OutcomeReplay:
    def __init__(self, capacity, seed, feature_dim=0, expert_success_fraction=0., obs_dim=50):
        if expert_success_fraction not in (0., .2):
            raise ValueError('Supported success expert fractions: 0 or 0.2')
        self.capacity = capacity
        self.seed = seed
        self.expert_success_fraction = expert_success_fraction
        self.expert_success = None
        self.success = Replay(capacity * 7 // 10, seed, feature_dim, obs_dim)
        self.failure = Replay(capacity - self.success.capacity, seed + 1, feature_dim, obs_dim)
        self.rng = np.random.default_rng(seed + 2)
        self.last_sample_counts = {}
        self.skipped_incomplete = 0

    @property
    def online(self):
        # Compatibility with continuation/status code; both buckets are mutable.
        return self

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.expert_success is not None:
            for name in ('obs','actions','rewards','next_obs','terminal','critic_features','next_critic_features'):
                array = getattr(self.expert_success, name, None)
                if array is not None:
                    array.flags.writeable = False

    @property
    def count(self):
        return self.success.count + self.failure.count + self.expert_count

    @property
    def expert_count(self):
        return self.expert_success.count if self.expert_success is not None else 0

    def add_online(self, obs, action, reward, nxt, terminal, *, mode=None,
                   cycle_outcome=None, **kwargs):
        if cycle_outcome == 'incomplete':
            self.skipped_incomplete += 1
            return
        if cycle_outcome not in ('success', 'failure'):
            raise ValueError('Replay requires a finalized gait-cycle outcome')
        if mode not in (0, 2):
            raise ValueError('V34 only accepts Turn/Walk')
        if terminal and cycle_outcome != 'failure':
            raise ValueError('Fallen transition cannot be successful')
        getattr(self, cycle_outcome).add(obs, action, reward, nxt, terminal, **kwargs)

    def counts_for_batch(self, n):
        failures = min(int(n * .3 + .5), self.failure.count)
        return n - failures, failures

    def ready(self, n, **kwargs):
        successes, _ = self.counts_for_batch(n)
        experts = int(successes * self.expert_success_fraction + .5)
        return self.expert_count >= experts and self.success.count >= successes - experts

    def sample(self, n, **kwargs):
        if not self.ready(n):
            raise ValueError('Not enough successful transitions to fill batch')
        counts = self.counts_batch = self.counts_for_batch(n)
        successes, failures = counts
        experts = int(successes * self.expert_success_fraction + .5)
        online = successes - experts
        self.last_sample_counts = dict(success=successes, failure=failures,
                                      expert_success=experts, online_success=online)
        names = ['obs', 'actions', 'rewards', 'next_obs', 'terminal']
        if self.success.critic_feature_dim:
            names += ['critic_features', 'next_critic_features']
        batches = []
        for bucket, count in ((self.expert_success, experts), (self.success, online), (self.failure, failures)):
            if count == 0:
                continue
            indices = self.rng.choice(bucket.count, count, replace=False)
            batches.append(tuple(torch.from_numpy(getattr(bucket, k)[indices]) for k in names))
        order = torch.from_numpy(self.rng.permutation(n))
        return tuple(torch.cat(parts)[order] for parts in zip(*batches))

    def seed_experts(self, episodes, *, turn_only=False, turn_sign=None):
        """Only retain whole observed cycles; never join episode boundaries."""
        if self.expert_count or self.success.count or self.failure.count:
            raise ValueError('Seed experts only once, before online insertion')
        if turn_sign not in (None, -1., 1.):
            raise ValueError('Expert Turn sign must be None or +/-1')
        accepted = []
        allowed_modes = (0,) if turn_only or turn_sign is not None else (0, 2)
        for episode in episodes:
            a = episode['arrays']
            pending = []
            complete_start = False
            for i, reward in enumerate(episode['revalued_rewards']):
                obs, nxt = a['observations'][i], a['next_observations'][i]
                phase = (np.arctan2(obs[9], obs[8]) / (2*np.pi)) % 1
                next_phase = (np.arctan2(nxt[9], nxt[8]) / (2*np.pi)) % 1
                if not pending and min(phase, 1-phase) < 1e-6:
                    complete_start = True
                pending.append((i, reward))
                if next_phase >= phase:
                    continue
                valid = complete_start and all(
                    int(a['modes'][j]) in allowed_modes and not a['terminals'][j]
                    and not np.allclose(a['next_observations'][j][5:8], 0.)
                    and (turn_sign is None or float(a['observations'][j,7])*turn_sign > 0)
                    for j, _ in pending)
                if valid:
                    for j, r in pending:
                        accepted.append((a, j, r))
                pending = []
                complete_start = True
        if self.expert_success_fraction:
            budget = self.capacity * 7 // 10
            if not 0 < len(accepted) < budget:
                raise ValueError('Expert pool must fit within success capacity, leaving online space')
            feature_dim = self.success.critic_feature_dim
            obs_dim = self.success.obs.shape[1]
            self.expert_success = Replay(len(accepted), self.seed + 3, feature_dim, obs_dim)
            self.success = Replay(budget - len(accepted), self.seed, feature_dim, obs_dim)
            for a, j, r in accepted:
                self.expert_success.add(a['observations'][j], a['executed_actions'][j], r,
                    a['next_observations'][j], False, critic_features=a['critic_features'][j],
                    next_critic_features=a['next_critic_features'][j])
            # Fixed pool is read-only after construction; online insertion never targets it.
            for name in ('obs','actions','rewards','next_obs','terminal','critic_features','next_critic_features'):
                array = getattr(self.expert_success, name, None)
                if array is not None:
                    array.flags.writeable = False
        else:
            for a, j, r in accepted:
                self.add_online(a['observations'][j], a['executed_actions'][j], r,
                    a['next_observations'][j], False, mode=int(a['modes'][j]),
                    cycle_outcome='success', critic_features=a['critic_features'][j],
                    next_critic_features=a['next_critic_features'][j])
