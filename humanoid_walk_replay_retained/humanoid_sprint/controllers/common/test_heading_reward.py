import math
import unittest
from types import SimpleNamespace
from contracts import ActorObservation, CanonicalAction, OBSERVATION_FIELDS
from envs.nao_locomotion_env import NaoLocomotionEnv, EnvStep
from tasks.forward_walk_heading_v3 import ForwardWalkHeadingV3
from tasks.forward_walk_v1 import ForwardWalkTaskConfig


class HeadingRewardTests(unittest.TestCase):
    def reward(self, yaw, distance=.003, fell=False):
        values = [0.] * 18
        values[OBSERVATION_FIELDS.index('yaw')] = yaw
        before = ActorObservation(tuple([0.] * 18))
        after = ActorObservation(tuple(values))
        action = CanonicalAction(tuple([0.] * 10))
        step = EnvStep(before, action, after, distance, fell, False, dict(fell=fell))
        host = SimpleNamespace(task=ForwardWalkHeadingV3(ForwardWalkTaskConfig(heading_reward_weight=.1)))
        transition = NaoLocomotionEnv.step_transition(host, step, action.values)
        self.assertEqual(transition.next_observation, after)
        self.assertEqual(transition.terminated, fell)
        return transition.reward

    def test_heading_symmetry_periodicity_and_forward_reward(self):
        self.assertAlmostEqual(self.reward(0), .03)
        self.assertAlmostEqual(self.reward(math.pi/2), -.07)
        self.assertAlmostEqual(self.reward(-math.pi/2), -.07)
        self.assertAlmostEqual(self.reward(2*math.pi), .03)

    def test_no_stationary_bonus_and_fall_unchanged(self):
        self.assertEqual(self.reward(0, 0), 0)
        for yaw in [0, math.pi/2, math.pi]:
            self.assertEqual(self.reward(yaw, fell=True), -3)
