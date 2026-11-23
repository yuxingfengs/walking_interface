"""Offline checks for NAO joint initialization and standing validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

SIM = Path(__file__).resolve().parents[1]
sys.path.append(str(SIM))
SAC_ISAAC = SIM.parents[1]/'sac_isaac_lab'
sys.path.append(str(SAC_ISAAC/'training'))


class InitialPoseTests(unittest.TestCase):
    def test_ddpg_standing_pose_covers_the_entire_articulation(self):
        import xml.etree.ElementTree as ET
        from joint_initialization import validate_initial_pose
        asset = SIM/'assets/nao.urdf'
        positions = json.loads((SIM/'assets/ddpg_standing_pose.json').read_text(encoding='utf-8'))['joint_positions_rad']
        joints = [j for j in ET.parse(asset).getroot().findall('joint')
                  if j.get('type') in ('revolute', 'continuous')]
        self.assertEqual(set(positions), {j.get('name') for j in joints})
        validate_initial_pose(asset, positions)
        for joint in joints:
            rule = joint.find('mimic')
            if rule is not None:
                expected = positions[rule.get('joint')]*float(rule.get('multiplier', '1'))
                self.assertAlmostEqual(positions[joint.get('name')], expected+float(rule.get('offset', '0')))

    def test_standing_check_records_both_thresholds_and_retains_failed_trial(self):
        import csv
        import math
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from standing_check import check_standing
        from basewalk.core import JOINTS

        class Fell(RuntimeError):
            pass

        for falls, recovers in ((False, False), (True, False), (True, True)):
            with self.subTest(falls=falls, recovers=recovers), tempfile.TemporaryDirectory() as directory:
                env = SimpleNamespace(dt=.04, settings=dict(spawn_height=.306172, hip_tolerance_rad=.03,
                                     physics_dt=.02, decimation=2), q=np.arange(len(JOINTS)))
                def reset(command, *, observe_settle):
                    env.pos, env.gravity = [0., 0., .306172], [0., 0., -1.]
                    observe_settle(0, None)
                    if falls:
                        env.pos[2], env.gravity[2] = .19, -math.cos(1.1)
                    observe_settle(1, np.array([[0, 0, 20], [0, 0, 22]]))
                    if recovers:
                        env.pos[2], env.gravity[2] = .31, -1.
                        observe_settle(2, np.array([[0, 0, 20], [0, 0, 22]]))
                    elif falls:
                        raise Fell('upright check failed')
                env.reset = reset
                cfg = dict(environment=dict(floor_z=0., fall_height=.2, fall_tilt_rad=1.))
                with patch.dict(sys.modules, environment=SimpleNamespace(UprightResetError=Fell)):
                    result = check_standing(env, cfg, 'baseline', Path(directory))
                self.assertEqual(result['upright'], not falls)
                self.assertEqual(result['final_reset_passed'], not falls or recovers)
                self.assertEqual(result['final']['below_height'], falls and not recovers)
                self.assertEqual(result['final']['over_tilt'], falls and not recovers)
                self.assertEqual(result['first_threshold_crossing'] is not None, falls)
                with (Path(directory)/'standing.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                    first, last = rows[0], rows[-1]
                self.assertEqual(first['left_net_fz_n'], '')
                self.assertEqual(float(last['right_net_fz_n']), 22.)
                self.assertEqual(float(last[JOINTS[-1]]), len(JOINTS)-1)

    def test_approved_elbow_pose_and_source_derived_finger_ranges(self):
        from joint_initialization import derive_mimic_limits, validate_initial_pose
        from basewalk.core import JOINTS, DEFAULT_POSE
        source = SIM/'assets/source_nao.urdf'
        before = source.read_bytes()
        settings = json.loads((SIM/'setup/simulator_config.json').read_text(encoding='utf-8-sig'))
        pose = dict(zip(JOINTS, map(float, DEFAULT_POSE)))
        pose.update(settings['upper_joint_positions'])
        self.assertEqual(pose['LElbowRoll'], -.04)
        self.assertEqual(pose['RElbowRoll'], .04)
        validate_initial_pose(source, pose)
        rules = derive_mimic_limits(source)
        self.assertEqual(len(rules), 16)
        for rule in rules.values():
            self.assertIn(rule['master'], ('LHand','RHand'))
            self.assertAlmostEqual(rule['lower_rad'], 0.)
            self.assertAlmostEqual(rule['upper_rad'], .999899)
        pose['LElbowRoll'] = 0.
        with self.assertRaisesRegex(ValueError, 'LElbowRoll'):
            validate_initial_pose(source, pose)
        self.assertEqual(source.read_bytes(), before)

    def test_mimic_bounds_support_negative_multiplier_and_detect_invalid_graphs(self):
        from joint_initialization import derive_mimic_limits
        with tempfile.TemporaryDirectory() as location:
            path = Path(location)/'fixture.urdf'
            text = ('<robot name="NaoH25V50"><joint name="master" type="revolute">'
                    '<limit lower="-1" upper="2"/></joint><joint name="follower" type="continuous">'
                    '<mimic joint="master" multiplier="-2" offset="0.25"/></joint></robot>')
            path.write_text(text)
            rule = derive_mimic_limits(path)['follower']
            self.assertEqual((rule['lower_rad'],rule['upper_rad']), (-3.75,2.25))
            for reference in ('missing', 'follower'):
                path.write_text(text.replace('joint="master"','joint="'+reference+'"'))
                with self.assertRaises(ValueError):
                    derive_mimic_limits(path)

if __name__ == '__main__':
    unittest.main()
