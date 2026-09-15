import unittest
from dataclasses import replace
import numpy as np
from com_estimator import NaoCOMEstimator
from contracts import COMObservation
from tasks.forward_walk_v1 import ForwardWalkTaskV1, ForwardWalkTaskConfig
from test_unified_env import raw_state


class COMTests(unittest.TestCase):
    def setUp(self):
        self.model = NaoCOMEstimator()
        self.q = dict.fromkeys(self.model.joint_names, 0.)

    def test_straight_pose_geometry_mass_and_symmetry(self):
        com, soles = self.model.kinematics(self.q)
        self.assertAlmostEqual(self.model.total_mass, 5.30535)
        np.testing.assert_allclose(soles, [[0, .05, -.33301], [0, -.05, -.33301]], atol=1e-5)
        self.assertAlmostEqual(com[1], 0., places=10)
        xyz, valid = self.model.estimate(self.q, 0, 0, (1, 1))
        self.assertEqual(valid, 1.)
        self.assertTrue(.25 < xyz[2] < .35)

    def test_support_switch_and_flight(self):
        left, _ = self.model.estimate(self.q, 0, 0, (1, 0))
        right, _ = self.model.estimate(self.q, 0, 0, (0, 1))
        self.assertAlmostEqual(right[1] - left[1], .1)
        self.assertEqual(self.model.estimate(self.q, 0, 0, (0, 0)), ((0., 0., 0.), 0.))

    def test_all_body_encoders_and_tilt_affect_estimate(self):
        initial, _ = self.model.estimate(self.q, 0, 0, (1, 1))
        self.q['LShoulderPitch'] = 1.
        moved, _ = self.model.estimate(self.q, 0, 0, (1, 1))
        self.assertGreater(np.linalg.norm(np.array(initial)-moved), .001)
        tilted, _ = self.model.estimate(self.q, .2, 0, (1, 1))
        self.assertAlmostEqual(np.linalg.norm(tilted), np.linalg.norm(moved))
        del self.q['HeadPitch']
        with self.assertRaises(ValueError):
            self.model.estimate(self.q, 0, 0, (1, 1))

    def test_observation_has_no_world_position_or_height(self):
        task = ForwardWalkTaskV1(ForwardWalkTaskConfig())
        state = replace(raw_state(), estimated_com=(.01, -.02, .3), com_valid=1., foot_contacts=(1., 0.))
        obs = task.reset(state)
        self.assertIsInstance(obs, COMObservation)
        self.assertEqual(obs.values[-4:], (.01, -.02, .3, 1.))
        self.assertEqual(obs, task.observe(replace(state, position=(9, 8, 7), center_of_mass_height=7), 0))
