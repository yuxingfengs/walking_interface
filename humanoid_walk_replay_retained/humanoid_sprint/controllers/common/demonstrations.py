"""Schema helpers for aligned Behavior Cloning demonstrations."""

from dataclasses import dataclass
from typing import Any, Dict

from contracts import ActorObservation, CanonicalAction, contract_metadata
from expert_policy import ExpertMetadata


@dataclass(frozen=True)
class DemonstrationRecord:
    """An aligned ActorObservation_t -> CanonicalAction_t sample."""

    observation: ActorObservation
    action: CanonicalAction
    step: int
    sim_time: float
    position: tuple

    def to_row(self, demo_name: str, expert: ExpertMetadata) -> Dict[str, Any]:
        x, y, z = self.position
        return {
            "demo_name": demo_name,
            "expert_id": expert.expert_id,
            "uses_privileged_state": expert.uses_privileged_state,
            **contract_metadata(),
            "step": self.step,
            "sim_time": self.sim_time,
            "phase": self.observation.values[0],
            "x": x,
            "y": y,
            "z": z,
            **{
                f"state_{index}": value
                for index, value in enumerate(self.observation.values)
            },
            **{
                f"action_{index}": value
                for index, value in enumerate(self.action.values)
            },
        }

