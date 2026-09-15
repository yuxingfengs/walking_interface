import unittest
from eval_stall import EvalStallMonitor
from configs.project_config import CONFIG
from configs.runtime import validate
import copy


class EvalStallTests(unittest.TestCase):
    def test_exceeds_40_not_at_40(self):
        monitor = EvalStallMonitor(-5.)
        for _ in range(40 * 35):
            self.assertFalse(monitor.observe(-5.))
        self.assertEqual(monitor.stall_cycles, 40)
        for _ in range(34):
            self.assertFalse(monitor.observe(-5.))
        self.assertTrue(monitor.observe(-5.))

    def test_progress_clears_consecutive_count_in_both_directions(self):
        for direction in (-1, 1):
            monitor = EvalStallMonitor(0, window_steps=2, cycle_limit=1)
            monitor.observe(0); monitor.observe(0)
            monitor.observe(direction*.02)
            self.assertFalse(monitor.observe(direction*.04))
            self.assertEqual(monitor.stall_cycles, 0)

    def test_disabled_for_old_snapshots(self):
        config = copy.deepcopy(CONFIG)
        for key in ('stall_cycle_limit', 'stall_window_steps', 'stall_max_abs_x_progress_m'):
            config['evaluation'].pop(key)
        self.assertEqual(validate(config)['evaluation']['stall_cycle_limit'], 0)
        monitor = EvalStallMonitor(0, cycle_limit=0)
        self.assertFalse(any(monitor.observe(0) for _ in range(2000)))

    def test_new_model_starts_with_fresh_count(self):
        first = EvalStallMonitor(0, window_steps=1, cycle_limit=1)
        first.observe(0)
        self.assertTrue(first.observe(0))
        self.assertFalse(EvalStallMonitor(0, window_steps=1, cycle_limit=1).observe(0))
