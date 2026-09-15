"""Forward walking with distance and fall rewards only."""
from tasks.forward_walk_v1 import ForwardWalkTaskV1


class ForwardWalkDistanceV2(ForwardWalkTaskV1):
    # Observation/action contracts stay compatible with the original BC Actor.
    reward_version = 'distance_fall_step_v2'

    @staticmethod
    def step_reward(action, previous_action, distance_moved, fell):
        move = max(min(0.5, distance_moved * 10.0), -0.5)
        return max(min(move - (5.0 if fell else 0.0), 3.0), -3.0)

    @staticmethod
    def reward(action, previous_action, distance_moved, fell, rollout_length, step_index):
        return ForwardWalkDistanceV2.step_reward(action, previous_action, distance_moved, fell)
