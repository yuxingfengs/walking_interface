"""Online FIFO insertion and immutable expert sampling are separate interfaces."""
import numpy as np
import torch
from .learning import Replay


class FixedExpertStore:
    def __init__(self, groups, capacities, seed, feature_dim):
        self._buffers = []
        for index, (rows, capacity) in enumerate(zip(groups, capacities)):
            if (not rows and index != 1) or len(rows) > capacity:
                raise ValueError('Expert data must be nonempty and fit protected capacity')
            buffer = Replay(max(1, len(rows)), seed+index, feature_dim)
            for mode, obs, action, reward, nxt, terminal, features, next_features in rows:
                buffer.add(obs, action, reward, nxt, terminal,
                           critic_features=features, next_critic_features=next_features)
            for value in vars(buffer).values():
                if isinstance(value, np.ndarray): value.flags.writeable = False
            self._buffers.append(buffer)

    @property
    def counts(self):
        return tuple(b.count for b in self._buffers)

    def sample(self, *counts):
        if len(counts) != 3 or len(self._buffers) != 3:
            raise ValueError('Expected Turn/Stand/Walk expert quotas')
        return tuple(torch.cat(parts) for parts in zip(*(b.sample(n) for b,n in zip(self._buffers,counts))))


class OnlineExpertReplay:
    """Only add_online writes data; experts have no episode admission API."""
    def __init__(self, capacity, experts, ratios, seed, feature_dim, turn_capacity=0, turn_fraction=0.):
        if not 0 < turn_capacity < capacity or not 0 < turn_fraction < ratios[0]:
            raise ValueError("Invalid extra Turn Replay capacity/fraction")
        self.online_turn = Replay(turn_capacity, seed+11, feature_dim)
        self.turn_fraction = turn_fraction
        self.online = Replay(capacity-turn_capacity, seed, feature_dim)
        self.experts, self.ratios = experts, ratios
        self.rng = np.random.default_rng(seed+3)
        self.last_sample_counts = {}

    @property
    def count(self):
        return self.online.count + sum(self.experts.counts)

    def add_online(self, obs, action, reward, nxt, terminal, *, mode,
                   critic_features=None, next_critic_features=None):
        if mode not in (0, 2):
            raise ValueError("Online source mode must be Turn0 or Walk2")
        if mode == 0:
            self.online_turn.add(obs, action, reward, nxt, terminal, critic_features=critic_features, next_critic_features=next_critic_features)
        self.online.add(obs, action, reward, nxt, terminal,
                        critic_features=critic_features, next_critic_features=next_critic_features)

    def ready(self, n, critic_only=False):
        return self.online.count >= n and self.online_turn.count > 0

    def sample(self, n, use_expert=True, critic_only=False):
        if not use_expert or not self.ready(n):
            raise ValueError('Mixed Replay requires fixed expert quotas and enough online data')
        online, extra_turn, turn, stand, walk = quotas((round(self.ratios[0]-self.turn_fraction, 12), self.turn_fraction, *self.ratios[1:]),n,self.rng)
        batches = (self.online.sample(online), self.online_turn.sample(extra_turn), self.experts.sample(turn,stand,walk))
        order = torch.from_numpy(self.rng.permutation(n))
        self.last_sample_counts = dict(online_mixed=online,online_turn=extra_turn,expert_turn=turn,expert_stand=stand,expert_walk=walk)
        return tuple(torch.cat(parts)[order] for parts in zip(*batches))


def quotas(ratios, n, rng):
    exact=np.asarray(ratios,dtype=float)*n
    counts=np.floor(exact).astype(int)
    left=n-int(counts.sum())
    if left:
        residual=exact-counts
        counts+=rng.multinomial(left,residual/residual.sum())
    return counts.tolist()
