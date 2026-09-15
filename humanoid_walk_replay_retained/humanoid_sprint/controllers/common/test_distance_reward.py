import unittest
from tasks.forward_walk_distance_v2 import ForwardWalkDistanceV2
from tasks.forward_walk_v1 import ForwardWalkTaskV1


class DistanceRewardTests(unittest.TestCase):
    def test_action_change_does_not_affect_reward(self):
        for action in ([0.] * 10, [2.] * 10):
            self.assertAlmostEqual(ForwardWalkDistanceV2.step_reward(action, [0.] * 10, .003, False), .03)
        self.assertLess(ForwardWalkTaskV1.step_reward([2.] * 10, [0.] * 10, .003, False), .03)

    def test_distance_direction_clipping_and_fall(self):
        for distance, expected in [(0., 0.), (.003, .03), (-.003, -.03), (1., .5), (-1., -.5)]:
            self.assertAlmostEqual(ForwardWalkDistanceV2.step_reward([], [], distance, False), expected)
            self.assertEqual(ForwardWalkDistanceV2.step_reward([], [], distance, True), -3.)
