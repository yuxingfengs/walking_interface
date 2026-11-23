"""Relative heading estimator shared by simulation and future robot adapters."""
import math


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class HeadingEstimator:
    """Integrate bias-corrected yaw rate from a caller-provided sensor adapter."""
    def __init__(self, initial_heading=0., yaw_rate_bias=0.):
        self.reset(initial_heading, yaw_rate_bias)

    def reset(self, initial_heading=0., yaw_rate_bias=0.):
        if not all(math.isfinite(v) for v in (initial_heading, yaw_rate_bias)):
            raise ValueError('Heading estimator requires finite initialization')
        self.heading = wrap(initial_heading)
        self.bias = float(yaw_rate_bias)

    def update(self, measured_yaw_rate, dt):
        if not math.isfinite(measured_yaw_rate) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Heading estimator requires finite yaw rate and positive dt')
        corrected = float(measured_yaw_rate)-self.bias
        self.heading = wrap(self.heading+corrected*dt)
        return self.heading, corrected
