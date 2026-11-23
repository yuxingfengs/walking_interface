import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from scripts.tracking_report import build_report_data


def sample(step, *, robot=0, clock=None, episode=1, vx=0., vy=0., wz=0., actual=(0., 0., 0.)):
    return dict(step=step, rank=robot, local_step=clock or step, episode=episode,
                motion_phase='turn' if wz else 'walk', vx_cmd=vx, vy_cmd=vy, wz_cmd=wz,
                actual_vx=actual[0], actual_vy=actual[1], actual_wz=actual[2],
                fell='false', truncated='false', reward=1.)


class TrackingReportTests(unittest.TestCase):
    @patch('scripts.tracking_report.EARLY_STEPS', 2)
    @patch('scripts.tracking_report.LATE_STEPS', 1)
    def test_first_last_reduction(self):
        rows = [sample(1, actual=(2., 1., 0.)), sample(2, actual=(-4., 1., 0.)),
                sample(3, actual=(99., 99., 0.)), sample(4, actual=(1.5, 2., .1), wz=.2)]
        metrics = build_report_data(rows, {'training': {}}, 4)['change']['metrics']
        self.assertEqual(metrics['vx'], dict(early=3., late=1.5, reduction_pct=50.))
        self.assertEqual(metrics['vy']['reduction_pct'], -100.)
        self.assertIsNone(metrics['wz']['reduction_pct'])
        self.assertFalse(build_report_data(rows[:2], {'training': {}}, 2)['change']['metrics'])

    def test_interleaved_collectors_and_same_target_in_new_hold(self):
        cfg = {'training': {'walk_commands': {'hold_steps': 2}}}
        rows = [sample(1, clock=1, actual=(1., 0., 0.)),
                sample(3, clock=2, actual=(-1., 0., 0.)),
                sample(5, clock=3, actual=(6., 0., 0.)),
                sample(2, robot=1, clock=1, vy=.04),
                sample(4, robot=1, clock=2, episode=2, vy=.04)]
        data = build_report_data(rows, cfg, 5)
        first, second = data['robots'][0]
        self.assertEqual((first[:3], first[8]), ([1, 3, 2], 0.))
        self.assertEqual((second[:3], second[8]), ([5, 5, 1], 6.))
        self.assertEqual(len(data['robots'][1]), 2)
        self.assertAlmostEqual(data['windows'][0]['vx'], 1.6)

    def test_window_metric_is_mean_of_per_step_norms(self):
        rows = [sample(i, actual=(3., 4., 0.) if i % 2 else (-3., -4., 0.))
                for i in range(1, 1001)]
        rows[999]['fell'] = 'true'
        rows.append(sample(1001, episode=2, wz=.11, actual=(0., 0., -.11)))
        data = build_report_data(rows, {'training': {}}, 1001)
        a, b = data['windows']
        self.assertEqual((a['n'], a['vx'], a['vy'], a['xy'], a['falls']), (1000, 3., 4., 5., 1))
        self.assertEqual((b['start'], b['end'], b['n'], b['xy']), (1001, 1001, 1, 0.))
        self.assertAlmostEqual(b['wz'], .22)
        self.assertTrue(data['turn'])
        self.assertEqual(sum(a['coverage'].values()), 1000)

    def test_accepts_single_robot_log_and_preserves_each_ramp_value(self):
        rows = [sample(i, wz=i*.01) for i in range(1, 4)]
        for row in rows:
            del row['rank'], row['local_step']
        data = build_report_data(rows, {'training': {'yaw_ramp': {}}}, 3)
        self.assertEqual([s[7] for s in data['robots'][0]], [.01, .02, .03])

    def test_missing_and_duplicate_steps(self):
        with self.assertRaises(ValueError):
            build_report_data([sample(1)], {'training': {}}, 2)
        with self.assertRaises(ValueError):
            build_report_data([sample(1), sample(1)], {'training': {}}, 2)

