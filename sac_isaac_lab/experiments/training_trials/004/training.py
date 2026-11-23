"""Continuous yaw reversal schedule."""
import numpy as np


class YawRampStages:
    """Per-control-step periodic Turn command, including through episode resets."""
    stage = 'turn'

    def __init__(self, cfg, state=None):
        training = cfg['training']
        self.rate = training['online_turn_rate_rad_s']
        self.ramp = training['yaw_ramp']['quarter_ramp_steps']
        self.hold = training['yaw_ramp']['hold_steps']
        low, high = cfg['environment']['command_ranges'][2]
        if not training['online_turn_only'] or not 0 < self.rate <= min(-low, high):
            raise ValueError('Yaw ramp requires a valid bidirectional Turn range')
        if any(type(n) is not int or n < 1 for n in (self.ramp, self.hold)):
            raise ValueError('Yaw ramp and hold durations require positive step counts')
        self.count = state['ramp_step'] if state is not None else 0
        self.period = 4 * (self.hold + self.ramp)
        if type(self.count) is not int or not 0 <= self.count < self.period:
            raise ValueError('Invalid yaw ramp continuation')

    def state(self):
        return dict(ramp_step=self.count)

    def command(self, error, speed):
        segment, offset = divmod(self.count, self.hold + self.ramp)
        levels = (1, 0, -1, 0, 1)
        fraction = max(0, offset - self.hold + 1) / self.ramp
        level = levels[segment] + fraction * (levels[segment + 1] - levels[segment])
        return np.array([0, 0, self.rate * level], np.float32)

    def advance_phase(self, before, after, *, fell=False):
        self.count = (self.count + 1) % self.period


command_schedule = YawRampStages
