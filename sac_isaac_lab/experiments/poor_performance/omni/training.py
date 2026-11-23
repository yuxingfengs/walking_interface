"""Alternate walk blocks with ramped turns."""
import numpy as np
from basewalk.omni_stages import OmniStages


class ExperimentSchedule(OmniStages):
    def __init__(self, cfg, state=None):
        super().__init__(cfg, state)
        spec = cfg['training']['omni_curriculum']
        self.ramp = spec['turn_ramp_steps']
        if not 1 <= self.ramp <= self.block // 2:
            raise ValueError('Invalid omni turn ramp')

    def command(self, error, speed):
        block_index, offset = divmod(self.step, self.block)
        if block_index % 2 == 0:
            return self.walk_command(block_index, offset)
        sign = 1 if (block_index // 2) % 2 == 0 else -1
        fraction = min(1., offset / self.ramp, (self.block - 1 - offset) / self.ramp)
        yaw = sign * self.peak * fraction
        return np.array([0., 0., yaw], dtype=np.float32)


command_schedule = ExperimentSchedule
