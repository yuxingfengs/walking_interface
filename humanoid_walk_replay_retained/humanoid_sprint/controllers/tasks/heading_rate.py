"""Circular heading error and causal world-heading rate, independent contract v6."""
import math
from tasks.heading_correction import HeadingCorrectionConfig
from tasks.forward_walk_v1 import ForwardWalkTaskV1
from contracts import HeadingRateObservation


def angle_difference(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


class HeadingRateTask(ForwardWalkTaskV1):
    config_type = HeadingCorrectionConfig
    reward_version = 'distance_fall_heading_sincos_rate_v6'

    def __init__(self, config):
        super().__init__(config)
        self._heading_sample = None
        self.yaw_rate = 0.
        self.heading_error = 0.

    def reset(self, raw_state):
        self._heading_sample = None
        self.yaw_rate = 0.
        result = super().reset(raw_state)
        # Env advances once after reset; the first exposed observation has no history.
        self._heading_sample = None
        return result

    def observe(self, raw_state, step_index):
        yaw, time = raw_state.world_heading, raw_state.simulation_time
        if yaw is None or not math.isfinite(yaw) or not math.isfinite(time):
            raise ValueError('Heading/rate observation requires finite heading and time')
        if self._heading_sample is None:
            self.yaw_rate = 0.
        else:
            previous_yaw, previous_time = self._heading_sample
            dt = time-previous_time
            if dt < 0:
                raise ValueError('Time moved backwards without task reset')
            if dt > 0:
                self.yaw_rate = angle_difference(yaw, previous_yaw)/dt
            elif abs(angle_difference(yaw, previous_yaw)) > 1e-12:
                raise ValueError('Heading changed without time advancing')
        self._heading_sample = (yaw, time)
        self.heading_error = angle_difference(self.config.target_heading_rad, yaw)
        base = super().observe(raw_state, step_index)
        return HeadingRateObservation(base.values + (math.sin(self.heading_error),
                                    math.cos(self.heading_error), self.yaw_rate))

    def step_reward_from_transition(self, step, previous_action):
        values = step.next_observation.values
        error = math.atan2(values[24], values[25])
        move = max(min(10.*step.distance_moved, .5), -.5)
        return max(min(move-self.config.heading_reward_weight*abs(error)
                       -(5. if step.info['fell'] else 0.), 3.), -3.)
