"""Expert Policy interface for demonstration collection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
from typing import Optional, Sequence

from contracts import ActorObservation, RawRobotState


@dataclass(frozen=True)
class ExpertMetadata:
    expert_id: str
    uses_privileged_state: bool
    description: str = ""


class ExpertPolicy(ABC):
    """A simulator-independent expert that proposes an Action."""

    metadata: ExpertMetadata

    @abstractmethod
    def act(
        self,
        observation: ActorObservation,
        raw_state: Optional[RawRobotState] = None,
    ) -> Sequence[float]:
        """Return an Action candidate for the same observation timestamp."""


def load_expert_policy(module_name: str, class_name: str) -> ExpertPolicy:
    """Load a configured ExpertPolicy plugin without importing Webots APIs."""
    module = importlib.import_module(module_name)
    policy_class = getattr(module, class_name)
    policy = policy_class()
    if not isinstance(policy, ExpertPolicy):
        raise TypeError(
            f"{module_name}.{class_name} must inherit ExpertPolicy, "
            f"got {type(policy).__name__}"
        )
    if not getattr(policy, "metadata", None):
        raise ValueError(f"{module_name}.{class_name} must declare ExpertMetadata")
    return policy

