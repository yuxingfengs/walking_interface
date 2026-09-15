"""Second S5 Task plugin: reward and terminate along world X-."""

from dataclasses import dataclass

from tasks.forward_walk_v1 import ForwardWalkTaskConfig, ForwardWalkTaskV1


@dataclass(frozen=True)
class BackwardWalkTaskConfig(ForwardWalkTaskConfig):
    stop_x: float = -1.0


class BackwardWalkTaskV1(ForwardWalkTaskV1):
    """Uses the shared Observation/Action contracts while reversing progress."""

    task_id = "backward_walk_v1"
    config_type = BackwardWalkTaskConfig

    def distance_moved(self, raw_state) -> float:
        if self.previous_position is None:
            self.previous_position = raw_state.position
            return 0.0
        distance = self.previous_position[0] - raw_state.position[0]
        self.previous_position = raw_state.position
        return float(distance)

    def termination(self, raw_state):
        fell = raw_state.center_of_mass_height < self.config.fall_height
        reached_goal = raw_state.position[0] < self.config.stop_x
        return fell, reached_goal
