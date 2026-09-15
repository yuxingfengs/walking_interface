"""Behavior-equivalence tests for the S4 Task and unified Env."""

import math
from pathlib import Path
import sys
import unittest
import torch


CONTROLLERS_DIR = Path(__file__).resolve().parent.parent
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from action_adapter import ActionAdapter
from contracts import ACTION_DIM, RawRobotState, STATE_DIM
from envs.nao_locomotion_env import NaoLocomotionEnv
from tasks.forward_walk_v1 import ForwardWalkTaskConfig, ForwardWalkTaskV1


def raw_state(*, time=0.0, x=0.0, height=0.7, joint_offset=0.0):
    return RawRobotState(
        roll=0.1,
        yaw=-0.2,
        gyro_x=0.3,
        gyro_y=-0.4,
        center_of_mass_height=height,
        joint_positions=tuple(joint_offset + index / 100.0 for index in range(12)),
        position=(x, 0.0, height),
        simulation_time=time,
    )


def legacy_reward(action, previous_action, distance, fell, rollout_length, step_index):
    reward = max(min(0.5, distance * 10.0), -0.5)
    step_ratio = (step_index + 1) / float(rollout_length)
    if fell:
        if step_ratio < 0.8:
            reward -= 0.1
        else:
            maximum = math.exp(10.0 * 0.2) - 1.0
            growth = math.exp(10.0 * (step_ratio - 0.8)) - 1.0
            reward += -0.01 - 4.99 * growth / maximum
    action_diff = float(torch.sum(
        (torch.tensor(action, dtype=torch.float32)
         - torch.tensor(previous_action, dtype=torch.float32)) ** 2
    ).item())
    if step_index == 0:
        action_diff = 0.2 if action_diff > 3.5 else 0.0
    elif action_diff <= 0.2:
        action_diff = 0.0
    return max(min(reward - action_diff, 3.0), -3.0)


class FakeRuntime:
    def __init__(self):
        self.states = [
            raw_state(time=10.00, x=0.00),
            raw_state(time=10.04, x=0.00),
            raw_state(time=10.08, x=0.00),
            raw_state(time=10.12, x=0.03),
        ]
        self.index = 0
        self.adapter = ActionAdapter()

    def initialize(self):
        return None

    def hard_reset(self, settle_steps=64):
        self.index = 0
        return self.states[0]

    def step(self):
        self.index = min(self.index + 1, len(self.states) - 1)
        return 0

    def read_state(self):
        return self.states[self.index]

    def apply_action(self, action):
        return self.adapter.adapt(action)

    def get_position(self):
        return self.states[self.index].position


class UnifiedEnvTests(unittest.TestCase):
    def setUp(self):
        self.config = ForwardWalkTaskConfig(
            sequence_length=35,
            phase_mode="continuous_time",
            phase_frequency_hz=4.0,
            phase_reset_value=0.0,
            stop_x=4.7,
        )

    def test_observation_matches_s0_field_order_and_phase(self):
        task = ForwardWalkTaskV1(self.config)
        origin = raw_state(time=5.0)
        task.reset(origin)
        current = raw_state(time=5.08, joint_offset=0.5)
        observation = task.observe(current, 2)
        expected_phase = 0.08 * 4.0 / (2.0 * math.pi)
        self.assertEqual(len(observation.values), 24)
        self.assertAlmostEqual(observation.values[0], expected_phase)
        self.assertEqual(observation.values[1:5], (0.1, -0.2, 0.3, -0.4))
        self.assertEqual(observation.values[6:18], current.joint_positions)

    def test_reward_matches_pre_s4_formula(self):
        action = tuple(index / 10.0 for index in range(ACTION_DIM))
        previous = (0.0,) * ACTION_DIM
        task = ForwardWalkTaskV1(self.config)
        for fell, length, index in ((False, 35, 0), (True, 17, 3), (True, 17, 16)):
            actual = task.reward(action, previous, 0.012, fell, length, index)
            expected = legacy_reward(action, previous, 0.012, fell, length, index)
            self.assertAlmostEqual(actual, expected, places=12)

    def test_env_returns_aligned_canonical_transition(self):
        runtime = FakeRuntime()
        env = NaoLocomotionEnv(runtime, ForwardWalkTaskV1(self.config))
        env.reset()
        observation = env.begin_rollout()
        step = env.step((0.0,) * ACTION_DIM, 0)
        transition = env.finalize_transition(
            step,
            (0.0,) * ACTION_DIM,
            rollout_fell=False,
            rollout_length=1,
            step_index=0,
        )
        self.assertEqual(transition.observation, observation)
        self.assertEqual(transition.action.values, (0.0,) * ACTION_DIM)
        self.assertAlmostEqual(step.distance_moved, 0.03)
        self.assertEqual(transition.next_observation, step.next_observation)
        self.assertFalse(transition.terminated)
        self.assertFalse(transition.truncated)

    def test_reaching_goal_preserves_s0_non_terminal_replay_semantics(self):
        runtime = FakeRuntime()
        runtime.states[-1] = raw_state(time=10.12, x=4.8)
        env = NaoLocomotionEnv(runtime, ForwardWalkTaskV1(self.config))
        env.reset()
        env.begin_rollout()
        step = env.step((0.0,) * ACTION_DIM, 0)
        self.assertTrue(step.info["reached_goal"])
        self.assertFalse(step.info["fell"])
        self.assertFalse(step.terminated)


if __name__ == "__main__":
    unittest.main()
