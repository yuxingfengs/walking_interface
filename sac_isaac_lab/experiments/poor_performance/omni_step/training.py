"""Alternate walk blocks with stepped turns."""
import numpy as np
from basewalk.omni_stages import OmniStages


class ExperimentSchedule(OmniStages):
    def __init__(self, cfg, state=None):
        super().__init__(cfg, state)
        spec = cfg['training']['omni_curriculum']
        self.turn_speed_steps = spec['turn_speed_steps']
        self.increment = spec['turn_increment_rad_s']
        if (type(self.turn_speed_steps) is not int or self.turn_speed_steps < 1
                or self.block % self.turn_speed_steps or not 0 < self.increment <= self.peak):
            raise ValueError('Invalid Turn speed increments or hold duration')
        self.peak_level = round(self.peak / self.increment)
        if not np.isclose(self.peak_level * self.increment, self.peak, rtol=0., atol=1e-8):
            raise ValueError('Turn increments must reach the signed peak exactly')

    def command(self, error, speed):
        block_index, offset = divmod(self.step, self.block)
        if block_index % 2 == 0:
            return self.walk_command(block_index, offset)
        turn_steps = (block_index // 2) * self.block + offset
        level = turn_steps // self.turn_speed_steps
        peak = self.peak_level
        yaw = (peak - abs((level + peak) % (4 * peak) - 2 * peak)) * self.increment
        return np.array([0., 0., yaw], dtype=np.float32)


command_schedule = ExperimentSchedule
