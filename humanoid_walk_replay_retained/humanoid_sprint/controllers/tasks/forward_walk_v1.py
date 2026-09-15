"""Behavior-preserving extraction of the S0 forward-walk Task."""

from dataclasses import dataclass
import math
import torch

from contracts import ActorObservation, RawRobotState
from tasks.task_plugin import TaskPlugin


@dataclass(frozen=True)
class ForwardWalkTaskConfig:
    sequence_length: int = 35
    phase_mode: str = "continuous_time"
    phase_frequency_hz: float = 4.0
    phase_reset_value: float = 0.0
    fall_height: float = 0.6
    stop_x: float = 4.7
    heading_reward_weight: float = 0.0


class ForwardWalkTaskV1(TaskPlugin):
    task_id = "forward_walk_v1"
    config_type = ForwardWalkTaskConfig

    def __init__(self, config: ForwardWalkTaskConfig):
        self.config = config
        self.phase_start_time = 0.0
        self.phase_start = config.phase_reset_value
        self.previous_position = None

    def reset(self, raw_state: RawRobotState) -> ActorObservation:
        self.phase_start_time = raw_state.simulation_time
        self.phase_start = self.config.phase_reset_value
        self.previous_position = raw_state.position
        return self.observe(raw_state, 0)

    def sync_position(self, raw_state: RawRobotState) -> None:
        self.previous_position = raw_state.position

    def phase(self, raw_state: RawRobotState, step_index: int) -> float:
        if self.config.phase_mode == "continuous_time":
            elapsed = raw_state.simulation_time - self.phase_start_time
            return (
                self.phase_start
                + elapsed * self.config.phase_frequency_hz / (2.0 * math.pi)
            ) % 1.0
        return (
            step_index % self.config.sequence_length
        ) / self.config.sequence_length

    def observe(self, raw_state: RawRobotState, step_index: int) -> ActorObservation:
        cm_norm = 0.0  # Reserved legacy slot; this project always uses sensor/model COM.
        values = (
            self.phase(raw_state, step_index),
            raw_state.roll,
            raw_state.yaw,
            raw_state.gyro_x,
            raw_state.gyro_y,
            cm_norm,
            *raw_state.joint_positions,
        )
        from contracts import COMObservation
        return COMObservation(values + tuple(raw_state.foot_contacts) + tuple(raw_state.estimated_com) + (raw_state.com_valid,))

    def distance_moved(self, raw_state: RawRobotState) -> float:
        if self.previous_position is None:
            self.previous_position = raw_state.position
            return 0.0
        distance = raw_state.position[0] - self.previous_position[0]
        self.previous_position = raw_state.position
        return float(distance)

    def termination(self, raw_state: RawRobotState):
        fell = raw_state.center_of_mass_height < self.config.fall_height
        reached_goal = raw_state.position[0] > self.config.stop_x
        return fell, reached_goal

    @staticmethod
    def step_reward(action, previous_action, distance_moved, fell):
        """Continuous sampling: no collection-window index or future fall labels."""
        move = max(min(0.5, distance_moved * 10.0), -0.5)
        difference = float(torch.sum((torch.as_tensor(action, dtype=torch.float32)
                                      - torch.as_tensor(previous_action, dtype=torch.float32)) ** 2).item())
        smoothness = difference if difference > 0.2 else 0.0
        # Same effective terminal clipping as the old final fall step.
        return max(min(move - smoothness - (5.0 if fell else 0.0), 3.0), -3.0)

    @staticmethod
    def reward(action, previous_action, distance_moved, fell, rollout_length, step_index):
        """Exact S0 Reward Function, moved without coefficient changes."""
        reward = 0.0
        step_ratio = (step_index + 1) / float(rollout_length)
        move_reward = max(min(0.5, distance_moved * 10.0), -0.5)
        reward += move_reward
        if fell:
            if step_ratio < 0.8:
                done_penalty = -0.1
            else:
                growth_scale = 10.0
                max_growth = math.exp(growth_scale * 0.2) - 1.0
                growth = math.exp(growth_scale * (step_ratio - 0.8)) - 1.0
                scaled = growth / max_growth if max_growth > 0 else 1.0
                done_penalty = -0.01 - (4.99 * scaled)
        else:
            done_penalty = 0.0
        reward += done_penalty

        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        previous_tensor = torch.as_tensor(previous_action, dtype=torch.float32)
        action_diff = float(torch.sum((action_tensor - previous_tensor) ** 2).item())
        if step_index == 0:
            action_diff = 0.2 if action_diff > 3.5 else 0.0
        elif action_diff <= 0.2:
            action_diff = 0.0
        reward -= action_diff
        return max(min(reward, 3.0), -3.0)
