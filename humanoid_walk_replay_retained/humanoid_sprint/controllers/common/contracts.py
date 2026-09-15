"""Versioned data contracts shared by training, BC, expert rollout, and evaluation."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple
import warnings


TASK_ID = "forward_walk_v1"
OBSERVATION_ID = "nao_locomotion_observation_v1"
ACTION_ID = "nao_absolute_leg_joint_position_v1"

OBSERVATION_FIELDS: Tuple[str, ...] = (
    "phase", "roll", "yaw", "gyro_x", "gyro_y", "cm_norm",
    "RHipYawPitch", "LHipYawPitch", "RHipRoll", "LHipRoll",
    "RHipPitch", "LHipPitch", "RKneePitch", "LKneePitch",
    "RAnklePitch", "LAnklePitch", "RAnkleRoll", "LAnkleRoll",
)
OBSERVATION_JOINT_ORDER: Tuple[str, ...] = OBSERVATION_FIELDS[6:]
ACTION_JOINT_ORDER: Tuple[str, ...] = (
    "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
)
ACTION_MIN: Tuple[float, ...] = (
    -0.379, -1.774, -0.092, -1.189, -0.397,
    -0.738, -1.774, -0.092, -1.186, -0.769,
)
ACTION_MAX: Tuple[float, ...] = (
    0.790, 0.484, 2.113, 0.923, 0.769,
    0.450, 0.484, 2.113, 0.932, 0.380,
)
ALL_LEG_JOINT_ORDER: Tuple[str, ...] = (
    "RHipYawPitch", "LHipYawPitch", "RHipRoll", "LHipRoll",
    "RHipPitch", "LHipPitch", "RKneePitch", "LKneePitch",
    "RAnklePitch", "LAnklePitch", "RAnkleRoll", "LAnkleRoll",
)
STATE_DIM = len(OBSERVATION_FIELDS)
CONTACT_OBSERVATION_FIELDS = OBSERVATION_FIELDS + ('left_foot_contact', 'right_foot_contact')
ACTION_DIM = len(ACTION_JOINT_ORDER)


@dataclass(frozen=True)
class ObservationSpec:
    observation_id: str = OBSERVATION_ID
    fields: Tuple[str, ...] = OBSERVATION_FIELDS

    @property
    def dim(self) -> int:
        return len(self.fields)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str = ACTION_ID
    joint_order: Tuple[str, ...] = ACTION_JOINT_ORDER
    semantics: str = "absolute_joint_position_radians"
    minimum: Tuple[float, ...] = ACTION_MIN
    maximum: Tuple[float, ...] = ACTION_MAX

    @property
    def dim(self) -> int:
        return len(self.joint_order)


OBSERVATION_SPEC = ObservationSpec()
ACTION_SPEC = ActionSpec()


@dataclass(frozen=True)
class RawRobotState:
    """Raw Webots feedback used to construct a deployable observation."""

    roll: float
    yaw: float
    gyro_x: float
    gyro_y: float
    center_of_mass_height: float
    joint_positions: Tuple[float, ...]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    simulation_time: float = 0.0
    foot_contacts: Tuple[float, float] = (0.0, 0.0)
    estimated_com: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    com_valid: float = 0.0
    estimated_forward: float = 0.0
    odometry_valid: bool = False
    world_heading: Optional[float] = None  # Privileged diagnostic input; never silently fall back to IMU.


@dataclass(frozen=True)
class ActorObservation:
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        validate_vector(self.values, STATE_DIM, "ActorObservation")


@dataclass(frozen=True)
class ContactObservation(ActorObservation):
    def __post_init__(self) -> None:
        validate_vector(self.values, len(CONTACT_OBSERVATION_FIELDS), 'ContactObservation')


@dataclass(frozen=True)
class COMObservation(ActorObservation):
    def __post_init__(self):
        validate_vector(self.values, 24, 'COMObservation')


@dataclass(frozen=True)
class HeadingObservation(ActorObservation):
    def __post_init__(self):
        validate_vector(self.values, 25, 'HeadingObservation')


@dataclass(frozen=True)
class HeadingRateObservation(ActorObservation):
    def __post_init__(self):
        validate_vector(self.values, 27, 'HeadingRateObservation')


@dataclass(frozen=True)
class CanonicalAction:
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        validate_vector(self.values, ACTION_DIM, "CanonicalAction")


@dataclass(frozen=True)
class Transition:
    observation: ActorObservation
    action: CanonicalAction
    reward: float
    next_observation: ActorObservation
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        """Legacy-compatible episode boundary used by the current DDPG target."""
        return self.terminated or self.truncated


def validate_vector(values: Sequence[Any], expected_dim: int, label: str) -> None:
    if len(values) != expected_dim:
        raise ValueError(f"{label} dim mismatch: got {len(values)}, expected {expected_dim}")


def contract_metadata(*, task_id: str = TASK_ID, state_dim: int = STATE_DIM) -> dict:
    return {
        "task_id": task_id,
        "observation_id": ('nao_heading_sincos_rate_v6' if state_dim == 27 else 'nao_absolute_heading_observation_v5' if state_dim == 25 else 'nao_sensor_com_observation_v3' if state_dim == 24 else 'nao_imu_contact_observation_v2' if state_dim == 20 else OBSERVATION_ID),
        "action_id": ACTION_ID,
    }


def validate_contract_metadata(
    metadata: Optional[Mapping[str, Any]],
    *,
    artifact: str,
    allow_legacy: bool = True,
    expected_task_id: str = TASK_ID,
    expected_state_dim: int = STATE_DIM,
) -> bool:
    """Validate artifact IDs; return False for an accepted legacy artifact."""
    metadata = metadata or {}
    expected = contract_metadata(task_id=expected_task_id, state_dim=expected_state_dim)
    missing = [key for key in expected if not metadata.get(key)]
    if missing:
        if not allow_legacy:
            raise ValueError(f"{artifact} missing contract metadata: {', '.join(missing)}")
        warnings.warn(
            f"{artifact} has no complete contract metadata and is treated as legacy: "
            f"{', '.join(missing)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    mismatches = [
        f"{key}={metadata[key]!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata[key] != value
    ]
    if mismatches:
        raise ValueError(f"incompatible {artifact} contracts: {'; '.join(mismatches)}")
    return True
