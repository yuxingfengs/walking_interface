"""Regression tests for the S2 Expert Policy and demonstration pipeline."""

import math
from pathlib import Path
import sys
import unittest


CONTROLLERS_DIR = Path(__file__).resolve().parent.parent
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from action_adapter import ActionAdapter
from contracts import ACTION_MAX, ACTION_MIN, HeadingRateObservation
from demonstrations import DemonstrationRecord
from expert_policy import load_expert_policy
from experts.strong_heading import StrongHeadingExpertPolicy


def legacy_open_loop_action(phase):
    """Exact pre-S2 formula retained here as the compatibility oracle."""
    angle = phase * 2.0 * math.pi
    robot_height = 0.5
    shift_y = 0.3
    step_height = 0.4
    step_length = 0.2

    left_roll = math.sin(angle) * shift_y
    left_knee = (math.sin(angle) + 1.0) / 2.0 * step_height + robot_height
    left_hip_pitch = -left_knee / 2.0 + math.cos(angle) * step_length
    left_ankle_pitch = -left_knee / 2.0 - math.cos(angle) * step_length
    right_roll = left_roll
    right_knee = (math.sin(angle + math.pi) + 1.0) / 2.0 * step_height + robot_height
    right_hip_pitch = -right_knee / 2.0 + math.cos(angle + math.pi) * step_length
    right_ankle_pitch = -right_knee / 2.0 - math.cos(angle + math.pi) * step_length
    return (
        left_roll, left_hip_pitch, left_knee, left_ankle_pitch, -left_roll,
        right_roll, right_hip_pitch, right_knee, right_ankle_pitch, -right_roll,
    )


class ExpertInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.policy = StrongHeadingExpertPolicy()
        self.adapter = ActionAdapter()

    def observation_for_phase(self, phase):
        return HeadingRateObservation((phase,) + (0.0,) * 23 + (0.0, 1.0, 0.0))

    def test_open_loop_output_matches_pre_s2_formula(self):
        for phase in (0.0, 0.125, 0.25, 0.5, 0.875):
            actual = self.policy.act(self.observation_for_phase(phase))
            expected = legacy_open_loop_action(phase)
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_configured_expert_loader_enforces_interface(self):
        policy = load_expert_policy("experts.strong_heading", "StrongHeadingExpertPolicy")
        self.assertIsInstance(policy, StrongHeadingExpertPolicy)

    def test_action_adapter_clips_to_action_spec(self):
        candidate = tuple(lower - 1.0 for lower in ACTION_MIN)
        self.assertEqual(self.adapter.adapt(candidate).values, ACTION_MIN)
        candidate = tuple(upper + 1.0 for upper in ACTION_MAX)
        self.assertEqual(self.adapter.adapt(candidate).values, ACTION_MAX)

    def test_action_adapter_intersects_runtime_motor_limits(self):
        class FakeMotor:
            def __init__(self, minimum, maximum):
                self.minimum = minimum
                self.maximum = maximum

            def getMinPosition(self):
                return self.minimum

            def getMaxPosition(self):
                return self.maximum

        from contracts import ACTION_JOINT_ORDER
        motors = {
            name: FakeMotor(lower, upper)
            for name, lower, upper in zip(ACTION_JOINT_ORDER, ACTION_MIN, ACTION_MAX)
        }
        motors["RHipRoll"] = FakeMotor(ACTION_MIN[5], 0.449597)
        adapter = ActionAdapter.from_motors(motors)
        self.assertEqual(adapter.adapt(ACTION_MAX).values[5], 0.449597)

    def test_demonstration_row_keeps_observation_and_action_aligned(self):
        observation = self.observation_for_phase(0.25)
        action = self.adapter.adapt(self.policy.act(observation))
        row = DemonstrationRecord(
            observation=observation,
            action=action,
            step=7,
            sim_time=1.5,
            position=(1.0, 2.0, 3.0),
        ).to_row("test_demo", self.policy.metadata)
        self.assertEqual(row["phase"], row["state_0"])
        self.assertEqual(row["step"], 7)
        self.assertEqual(row["expert_id"], "strong_heading_stride_v1")
        self.assertFalse(row["uses_privileged_state"])
        self.assertEqual(
            tuple(row[f"action_{index}"] for index in range(len(action.values))),
            action.values,
        )


if __name__ == "__main__":
    unittest.main()
