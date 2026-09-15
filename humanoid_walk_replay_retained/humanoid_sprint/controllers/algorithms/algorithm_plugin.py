"""Simulator-independent interface implemented by learning Algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

from contracts import Transition


@dataclass(frozen=True)
class AlgorithmUpdate:
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    metrics: Mapping[str, object] = field(default_factory=dict)


class AlgorithmPlugin(ABC):
    """Minimal contract consumed by the Webots training orchestrator."""

    algorithm_id = "algorithm_plugin"

    @abstractmethod
    def act(self, observation):
        """Return one candidate Action for one ActorObservation."""

    @abstractmethod
    def observe(self, transition: Transition) -> None:
        """Store or otherwise consume one environment Transition."""

    @abstractmethod
    def update(self) -> AlgorithmUpdate:
        """Perform one configured learning update."""

    @property
    @abstractmethod
    def experience_size(self) -> int:
        """Number of transitions currently available to the Algorithm."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Minimum batch used by one update."""

    @abstractmethod
    def clear_experience(self) -> None:
        """Discard experience that is incompatible with a restored policy."""

    @abstractmethod
    def set_evaluation_mode(self) -> None:
        """Put the action-producing policy into deterministic evaluation mode."""

    @abstractmethod
    def save_checkpoint(self, filepath, metadata=None):
        """Save complete resumable Algorithm state."""

    @abstractmethod
    def load_checkpoint(self, filepath):
        """Restore complete Algorithm state and return artifact metadata."""

    @abstractmethod
    def load_actor_initialization(
        self,
        filepath,
        *,
        reset_optimizer: bool,
        actor_lr=None,
    ) -> bool:
        """Initialize a compatible Actor-style policy artifact."""
