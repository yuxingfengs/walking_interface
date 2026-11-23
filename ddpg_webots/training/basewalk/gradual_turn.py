"""Attempt-boundary Turn staircase with bounded endpoints and a zero-rate Turn."""
import math
import numpy as np


def source_motion_stage(command, cfg, stage=None):
    if cfg.get('training', {}).get('walk_commands'):
        return 'walk'
    if cfg.get('training', {}).get('turn_and_walk', {}).get('enabled'):
        if stage in ('turn', 'walk'):
            return stage
        if abs(command[1]) > 1e-8:
            return 'walk'
    if abs(command[0]) > 1e-8:
        return 'walk'
    if abs(command[2]) > 1e-8:
        return 'turn'
    if cfg.get('training', {}).get('gradual_turn', {}).get('enabled'):
        return 'turn'
    return 'stand'


class GradualTurnStages:
    def __init__(self, cfg, start_stage='turn', *, cycle_limit=None, state=None,
                 count_fall_as_attempt=False, turn_only=False, fixed_turn_sign=None):
        spec = cfg['training']['gradual_turn']
        if not spec['enabled'] or not turn_only or not count_fall_as_attempt or fixed_turn_sign is not None:
            raise ValueError('Gradual Turn requires Turn-only, falls counted and no fixed sign')
        maximum = cfg['training']['online_turn_rate_rad_s']
        intervals = spec['intervals_per_half_range']
        hold = spec['endpoint_attempts']
        if type(intervals) is not int or intervals <= 0 or type(hold) is not int or hold <= 0:
            raise ValueError('Positive integer staircase intervals/holds required')
        low, high = cfg['environment']['command_ranges'][2]
        if not math.isfinite(maximum) or not 0 < maximum <= min(-low, high):
            raise ValueError('Gradual Turn maximum outside command bounds')
        if not cfg['environment']['phase_runs_at_zero_command']:
            raise ValueError('Zero Turn must keep the gait phase running')
        self.rates = [maximum * i / intervals for i in range(intervals, -intervals-1, -1)]
        self.rates += [maximum * i / intervals for i in range(-intervals+1, intervals)]
        self.holds = [hold if abs(abs(r)-maximum) < 1e-12 else 1 for r in self.rates]
        self.cfg = cfg
        self.stage = 'turn'
        self.count = 0
        self.count_fall_as_attempt = True
        self.index = 0
        self.cycles = 0
        if state is not None:
            index, cycles = state.get('gradual_index'), state.get('cycles')
            if (state.get('stage') != 'turn' or type(index) is not int or
                    not 0 <= index < len(self.rates) or type(cycles) is not int or
                    not 0 <= cycles < self.holds[index]):
                raise ValueError('Invalid gradual Turn continuation state')
            self.index, self.cycles = index, cycles

    def command(self, error, speed):
        return np.array([0., 0., self.rates[self.index]], dtype=np.float32)

    @property
    def limit(self):
        return self.holds[self.index]

    def state(self):
        return dict(stage='turn', cycles=self.cycles, gradual_index=self.index,
                    turn_sign=-1. if self.rates[self.index] < 0 else 1.)

    def advance_phase(self, before, after, *, fell=False):
        if not fell and after >= before:
            return
        self.cycles += 1
        if self.cycles >= self.holds[self.index]:
            self.cycles = 0
            self.index = (self.index + 1) % len(self.rates)
