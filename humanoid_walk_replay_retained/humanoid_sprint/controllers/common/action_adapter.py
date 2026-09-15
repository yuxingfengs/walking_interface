"""Convert policy output into the CanonicalAction applied by Webots."""

from typing import Mapping, Sequence

from contracts import (
    ACTION_JOINT_ORDER,
    ACTION_MAX,
    ACTION_MIN,
    CanonicalAction,
    validate_vector,
)


class ActionAdapter:
    """Order-preserving clipping for ActionSpec v1 absolute joint targets."""

    def __init__(
        self,
        minimum: Sequence[float] = ACTION_MIN,
        maximum: Sequence[float] = ACTION_MAX,
    ):
        validate_vector(minimum, len(ACTION_MIN), "Action minimum")
        validate_vector(maximum, len(ACTION_MAX), "Action maximum")
        self.minimum = tuple(float(value) for value in minimum)
        self.maximum = tuple(float(value) for value in maximum)
        if any(lower > upper for lower, upper in zip(self.minimum, self.maximum)):
            raise ValueError("Action minimum exceeds maximum")

    @classmethod
    def from_motors(cls, motors: Mapping[str, object]) -> "ActionAdapter":
        """Intersect the contract bounds with the active Webots motor limits."""
        minimum, maximum = [], []
        for name, contract_min, contract_max in zip(
            ACTION_JOINT_ORDER, ACTION_MIN, ACTION_MAX
        ):
            motor = motors[name]
            motor_min = float(motor.getMinPosition())
            motor_max = float(motor.getMaxPosition())
            minimum.append(max(contract_min, motor_min))
            maximum.append(min(contract_max, motor_max))
        return cls(minimum, maximum)

    def adapt(self, candidate: Sequence[float]) -> CanonicalAction:
        validate_vector(candidate, len(ACTION_MIN), "Action candidate")
        values = tuple(
            min(max(float(value), lower), upper)
            for value, lower, upper in zip(candidate, self.minimum, self.maximum)
        )
        return CanonicalAction(values)

