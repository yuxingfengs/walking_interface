"""Alternating Walk and Turn command blocks on the persistent environment-step clock."""
import numpy as np


class WalkCommands:
    """Hold each body-frame XY target on a persistent per-robot step clock."""
    def __init__(self, cfg, state=None):
        spec = cfg['training']['walk_commands']
        self.random_vx_range = spec.get('random_vx_range')
        self.robot_index = (state or {}).get('robot_index', 0)
        self.vx_cycle = None
        self.hold = spec['hold_steps']
        xy = np.asarray(spec['commands_m_s'], dtype=np.float32)
        limits = np.asarray(cfg['environment']['command_ranges'][:2])
        if (xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all()
                or np.any(np.linalg.norm(xy, axis=1) == 0) or np.any(xy < limits[:, 0])
                or np.any(xy > limits[:, 1]) or type(self.hold) is not int or self.hold < 1):
            raise ValueError('Invalid Walk targets or hold duration')
        self.commands = np.column_stack((xy, np.zeros(len(xy), dtype=np.float32)))
        self.repeat_from = spec.get('repeat_from_index', 0)
        if type(self.repeat_from) is not int or not 0 <= self.repeat_from < len(xy):
            raise ValueError('Invalid first repeating Walk command index')
        self.step = 0 if state is None else state['step']
        if type(self.step) is not int or self.step < 0:
            raise ValueError('Invalid Walk command clock')

    def state(self):
        return {'step': self.step}

    def command(self, error, speed):
        index = self.step // self.hold
        if index >= len(self.commands):
            index = self.repeat_from + (index - self.repeat_from) % (len(self.commands) - self.repeat_from)
        cmd = self.commands[index].copy()
        if self.random_vx_range is not None:
            cycle = self.step // (self.hold * len(self.commands))
            if self.vx_cycle != cycle:
                rng = np.random.default_rng([self.robot_index, cycle, 42])
                self.vx = rng.uniform(*self.random_vx_range)
                self.vx_cycle = cycle
            cmd[0] = self.vx
        return cmd

    def advance_phase(self, before, after, *, fell=False):
        self.step += 1


class OmniStages:
    def __init__(self, cfg, state=None):
        spec = cfg['training']['omni_curriculum']
        self.block = spec['block_steps']
        self.walk_speed_steps = spec.get('walk_speed_steps', self.block)
        if type(self.walk_speed_steps) is not int or self.walk_speed_steps <= 0 or self.block % self.walk_speed_steps:
            raise ValueError('Walk speed duration must evenly divide its task block')
        self.ramp = spec.get('turn_ramp_steps')
        self.peak = spec['turn_peak_rad_s']
        if self.ramp is None:
            self.turn_speed_steps = spec['turn_speed_steps']
            self.increment = spec['turn_increment_rad_s']
            if (type(self.turn_speed_steps) is not int or self.turn_speed_steps < 1
                    or self.block % self.turn_speed_steps or not 0 < self.increment <= self.peak):
                raise ValueError('Invalid Turn speed increments or hold duration')
            self.peak_level = round(self.peak / self.increment)
            if not np.isclose(self.peak_level * self.increment, self.peak, rtol=0., atol=1e-8):
                raise ValueError('Turn increments must reach the signed peak exactly')
        self.walk_stages = spec['walk_stages']
        self.step = 0 if state is None else state['step']
        if self.step < 0 or (self.ramp is not None and not 1 <= self.ramp <= self.block // 2):
            raise ValueError('Invalid omni command clock')
        starts = [s['start_step'] for s in self.walk_stages]
        if starts[0] != 0 or sorted(set(starts)) != starts or any(i % (2*self.block) for i in starts):
            raise ValueError('Walk curriculum changes must start on Walk block boundaries')
        limits = np.asarray(cfg['environment']['command_ranges'])
        if not 0 < self.peak <= min(-limits[2, 0], limits[2, 1]):
            raise ValueError('Turn peak exceeds the task command range')
        for stage in self.walk_stages:
            xy = np.asarray(stage['commands_m_s'])
            if (xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all()
                    or np.any(np.linalg.norm(xy, axis=1) < 1e-6)
                    or np.any(xy < limits[:2, 0]) or np.any(xy > limits[:2, 1])):
                raise ValueError('Invalid planar Walk command')

    def state(self):
        return {'step': self.step}

    def command(self, error, speed):
        block_index, offset = divmod(self.step, self.block)
        if block_index % 2:
            if self.ramp is None:
                turn_steps = (block_index // 2) * self.block + offset
                level = turn_steps // self.turn_speed_steps
                # Triangle in integer levels, starting at zero and counting only Turn steps.
                peak = self.peak_level
                signed_level = peak - abs((level + peak) % (4 * peak) - 2 * peak)
                return np.array([0., 0., signed_level * self.increment], dtype=np.float32)
            # Each Turn starts and ends at zero, with a hold at the signed peak.
            sign = 1 if (block_index // 2) % 2 == 0 else -1
            fraction = min(1., offset / self.ramp, (self.block - 1 - offset) / self.ramp)
            return np.array([0., 0., sign * self.peak * fraction], dtype=np.float32)
        stage = next(s for s in reversed(self.walk_stages) if self.step >= s['start_step'])
        walk_block = (self.step - stage['start_step']) // (2 * self.block)
        index = walk_block * (self.block // self.walk_speed_steps) + offset // self.walk_speed_steps
        vx, vy = stage['commands_m_s'][index % len(stage['commands_m_s'])]
        return np.array([vx, vy, 0.], dtype=np.float32)

    def advance_phase(self, before, after, *, fell=False):
        self.step += 1
