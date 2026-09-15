"""Privileged absolute-heading diagnostic; preserves original 24 BC inputs."""
from dataclasses import dataclass
import math
from tasks.forward_walk_v1 import ForwardWalkTaskConfig, ForwardWalkTaskV1
from contracts import HeadingObservation


@dataclass(frozen=True)
class HeadingCorrectionConfig(ForwardWalkTaskConfig):
    target_heading_rad: float=0.


class HeadingCorrectionTask(ForwardWalkTaskV1):
    config_type=HeadingCorrectionConfig
    reward_version='distance_fall_world_heading_v5'

    def observe(self,raw_state,step_index):
        original=super().observe(raw_state,step_index)
        if raw_state.world_heading is None or not math.isfinite(raw_state.world_heading):
            raise ValueError('Absolute heading diagnostic requires finite world_heading')
        difference=self.config.target_heading_rad-raw_state.world_heading
        error=math.atan2(math.sin(difference),math.cos(difference))
        return HeadingObservation(original.values+(error,))

    def step_reward_from_transition(self,step,previous_action):
        move=max(min(10.*step.distance_moved,.5),-.5)
        heading=-self.config.heading_reward_weight*abs(step.next_observation.values[24])
        return max(min(move+heading-(5. if step.info['fell'] else 0.),3.),-3.)
