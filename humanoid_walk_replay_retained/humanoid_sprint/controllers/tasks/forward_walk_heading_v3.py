"""Distance/fall reward with world +x heading alignment."""
import math
from contracts import OBSERVATION_FIELDS
from tasks.forward_walk_distance_v2 import ForwardWalkDistanceV2


class ForwardWalkHeadingV3(ForwardWalkDistanceV2):
    reward_version = 'distance_fall_heading_v3'

    def step_reward_from_transition(self, step, previous_action):
        # Use the executed action's successor, never mutable state after a reset.
        yaw = step.next_observation.values[OBSERVATION_FIELDS.index('yaw')]
        move = max(min(0.5, step.distance_moved * 10.0), -0.5)
        heading = self.config.heading_reward_weight * (math.cos(yaw) - 1.0)
        return max(min(move + heading - (5.0 if step.info['fell'] else 0.0), 3.0), -3.0)
