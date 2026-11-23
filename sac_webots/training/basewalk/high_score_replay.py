"""Whole-cycle return-ranked source layered on the existing online Replay."""
from bisect import bisect_right
import heapq
import math
from pathlib import Path

import numpy as np
import torch

from .expert_pipeline import digest
from .outcome_replay import OutcomeReplay

FIELDS = ('obs', 'actions', 'rewards', 'next_obs', 'terminal')


class HighScoreReplay(OutcomeReplay):
    capacity_key = 'capacity_cycles'
    pool_size_log_field = 'replay_high_score_pool_cycles'

    def __init__(self, *args, spec, target_turn_rate, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_fraction = spec['batch_fraction']
        self.high_capacity = spec[self.capacity_key]
        self.target_turn_rate = target_turn_rate
        if not 0 < self.high_fraction <= 0.5 or type(self.high_capacity) is not int or self.high_capacity <= 0:
            raise ValueError('Invalid high-score Replay quota or capacity')
        if self.success.critic_feature_dim:
            raise ValueError('High-score Replay requires the shared Actor State contract')
        self.high_candidates = []
        self.high_selected = []
        self.high_ends = []
        self.success_steps = None

    def seed_high_score(self, path, sha256):
        path = Path(path)
        if digest(path) != sha256:
            raise ValueError('High-score Replay seed changed')
        if self.high_count:
            raise ValueError('High-score Replay may only be seeded once')
        with np.load(path, allow_pickle=False) as source:
            arrays = tuple(source[k] for k in FIELDS)
            offsets, ids = source['offsets'], source['ids']
            rates = source['net_rates']
        n = len(ids)
        expected = ((n,self.success.obs.shape[1]), (n,12), (n,1),
                    (n,self.success.obs.shape[1]), (n,1))
        if any(a.shape != shape or not np.isfinite(a).all() for a, shape in zip(arrays,expected)):
            raise ValueError('Invalid high-score Replay seed transitions')
        if (len(offsets) != len(rates)+1 or offsets[0] != 0 or offsets[-1] != n
                or np.any(np.diff(offsets) <= 0) or np.any(np.diff(ids) >= 0)
                or np.any(ids >= 0) or not np.isfinite(rates).all() or arrays[-1].any()):
            raise ValueError('Invalid high-score Replay seed cycle bounds')
        for i, rate in enumerate(rates):
            sl = slice(offsets[i], offsets[i+1])
            self._append_high_score(tuple(a[sl] for a in arrays), ids[sl], float(rate))
        self.success_steps = np.zeros(self.success.capacity, dtype=np.int64)
        self.refresh_high_score()

    def _append_high_score(self, arrays, ids, rate):
        error = abs(rate - self.target_turn_rate)
        reward = float(arrays[2].mean(dtype=np.float64))
        self.high_candidates.append((error, reward, ids, arrays))

    def refresh_high_score(self):
        # Recompute the eligible half from historical and newly collected
        # cycles. Every archived transition has its original State and Action.
        candidates = sorted(self.high_candidates, key=lambda item: item[0])
        eligible = candidates[:max(1, len(candidates)//2)]
        self.high_selected = sorted(eligible, key=lambda item: -item[1])[:self.high_capacity]
        self.high_ends = np.cumsum([len(item[2]) for item in self.high_selected]).tolist()

    def add_high_score_cycle(self, cycle):
        obs, action, rewards, nxt, terminal, rows = zip(*cycle)
        ids = np.asarray([r['step'] for r in rows], dtype=np.int64)
        if (any(terminal) or np.any(np.diff(ids) != 1)
                or any(r['motion_phase'] != 'turn' or abs(r['wz_cmd']-self.target_turn_rate) > 1e-7
                       for r in rows)):
            raise ValueError('High-score Replay requires an intact fixed-command Turn cycle')
        duration = math.fsum(r['cycle_dt'] for r in rows)
        if duration <= 0:
            raise ValueError('High-score Replay cycle duration must be positive')
        rate = math.fsum(r['yaw_delta'] for r in rows)/duration
        arrays = tuple(np.asarray(x, dtype=np.float32) for x in (obs, action, rewards, nxt, terminal))
        arrays = (*arrays[:2], arrays[2][:,None], arrays[3], arrays[4][:,None])
        # The intact cycle has just been appended to regular success Replay.
        slots = (self.success.cursor - len(ids) + np.arange(len(ids))) % self.success.capacity
        self.success_steps[slots] = ids
        self._append_high_score(arrays, ids, rate)
        self.refresh_high_score()

    @property
    def high_count(self):
        return self.high_ends[-1] if self.high_ends else 0

    def ready(self, n, **kwargs):
        return super().ready(n, **kwargs) and self.high_count >= int(n*self.high_fraction + .5)

    def sample(self, n, **kwargs):
        if not self.ready(n):
            raise ValueError('Not enough transitions to fill the high-score batch')
        successes, failures = self.counts_batch = self.counts_for_batch(n)
        experts = int(successes*self.expert_success_fraction + .5)
        high = int(n*self.high_fraction + .5)
        online = successes - experts - high
        self.last_sample_counts = dict(success=successes, failure=failures, expert_success=experts,
                                       online_success=online, high_score=high)
        high_ids = []
        selected = []
        for j in self.rng.choice(self.high_count, high, replace=False):
            i = bisect_right(self.high_ends, j)
            offset = j - (self.high_ends[i-1] if i else 0)
            _, _, ids, arrays = self.high_selected[i]
            high_ids.append(ids[offset])
            selected.append(tuple(a[offset] for a in arrays))
        batch = [tuple(torch.from_numpy(np.stack(column)) for column in zip(*selected))] if high else []
        allowed = np.flatnonzero(~np.isin(self.success_steps[:self.success.count], high_ids))
        for pool, candidates, count in (
                (self.expert_success, self.expert_count, experts),
                (self.success, allowed, online),
                (self.failure, self.failure.count, failures)):
            if count:
                idx = self.rng.choice(candidates, count, replace=False)
                batch.append(tuple(torch.from_numpy(getattr(pool, k)[idx]) for k in FIELDS))
        order = torch.from_numpy(self.rng.permutation(n))
        return tuple(torch.cat(columns)[order] for columns in zip(*batch))


class HighScoreTransitionReplay(HighScoreReplay):
    """Select individual successful samples by final per-step Reward."""
    capacity_key = 'capacity_transitions'
    pool_size_log_field = 'replay_high_score_pool_transitions'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_serial = 0

    def _append_high_score(self, arrays, ids, rate):
        for i, score in enumerate(arrays[2][:, 0]):
            self.high_serial += 1
            priority = (float(score), -self.high_serial)
            if len(self.high_candidates) >= self.high_capacity and priority <= self.high_candidates[0][:2]:
                continue
            one = slice(i, i+1)
            item = (*priority, ids[one].copy(), tuple(a[one].copy() for a in arrays))
            if len(self.high_candidates) < self.high_capacity:
                heapq.heappush(self.high_candidates, item)
            else:
                heapq.heapreplace(self.high_candidates, item)

    def refresh_high_score(self):
        self.high_selected = self.high_candidates
        if len(self.high_ends) != len(self.high_selected):
            self.high_ends = list(range(1, len(self.high_selected)+1))
