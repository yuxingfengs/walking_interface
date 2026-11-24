"""Cycle through lateral targets and draw one forward target per robot per cycle."""

# The forward component is replaced by the sampled vx once per command cycle.
commands_m_s = [[0.0873257964849472, vy] for vy in (0., .04, .08, .04, 0., -.04, -.08, -.04)]
hold_steps = 100
vx_range_m_s = [-.11, .11]


def command_schedule(cfg, state=None):
    import numpy as np
    from basewalk.omni_stages import WalkCommands

    class RandomForwardCommands(WalkCommands):
        def __init__(self, cfg, state=None):
            super().__init__(cfg, state)
            self.robot_index = (state or {}).get('robot_index', 0)
            self.cycle = None
            self.vx_range = cfg['environment']['command_ranges'][0]

        def command(self, error, speed):
            cmd = self.target_at(self.step // self.hold)
            cycle = self.step // (self.hold * len(self.commands))
            if self.cycle != cycle:
                rng = np.random.default_rng([self.robot_index, cycle, 42])
                self.vx = rng.uniform(*self.vx_range)
                self.cycle = cycle
            cmd[0] = self.vx
            return cmd

    return RandomForwardCommands(cfg, state)
