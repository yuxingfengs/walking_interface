"""Evaluation-only stall detection; observes positions without stepping simulation."""


class EvalStallMonitor:
    def __init__(self, start_x, *, window_steps=35, cycle_limit=40, max_abs_progress_m=.02):
        self.start_x = float(start_x)
        self.window_steps = window_steps
        self.cycle_limit = cycle_limit
        self.max_abs_progress_m = max_abs_progress_m
        self.steps = 0
        self.stall_cycles = 0

    def observe(self, x):
        if self.cycle_limit == 0:
            return False
        self.steps += 1
        if self.steps < self.window_steps:
            return False
        progress = abs(float(x) - self.start_x)
        self.stall_cycles = self.stall_cycles + 1 if progress <= self.max_abs_progress_m else 0
        self.start_x = float(x)
        self.steps = 0
        return self.stall_cycles > self.cycle_limit
