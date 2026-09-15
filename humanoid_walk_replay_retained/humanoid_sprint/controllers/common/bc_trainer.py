"""Simulator-independent Behavior Cloning trainer and artifact exporter."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from contracts import ACTION_DIM, STATE_DIM, contract_metadata
from demonstration_dataset import DemonstrationDataset
from models import BoundedActor


@dataclass(frozen=True)
class BCTrainingConfig:
    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-4
    seed: int = 1

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class BCTrainingResult:
    artifact_path: Path
    metadata_path: Path
    final_mse: float
    metadata: dict


class BCTrainer:
    def __init__(self, config: BCTrainingConfig, device: Optional[torch.device] = None):
        config.validate()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, dataset: DemonstrationDataset, output_path: Path) -> BCTrainingResult:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        actor = BoundedActor(STATE_DIM, 256, ACTION_DIM).to(self.device)
        actions = torch.maximum(
            torch.minimum(dataset.actions, actor.joint_max.cpu()),
            actor.joint_min.cpu(),
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(dataset.states, actions),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(actor.parameters(), lr=self.config.learning_rate)

        actor.train()
        final_loss = None
        for epoch in range(1, self.config.epochs + 1):
            total_loss = 0.0
            total_samples = 0
            for batch_states, batch_actions in loader:
                predictions = actor(batch_states.to(self.device))
                loss = functional.mse_loss(predictions, batch_actions.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_states)
                total_samples += len(batch_states)
            final_loss = total_loss / max(total_samples, 1)
            if epoch == 1 or epoch % 25 == 0 or epoch == self.config.epochs:
                print(f"[bc-trainer] epoch={epoch:04d} loss={final_loss:.8f}")

        metadata = {
            **contract_metadata(),
            "artifact_type": "bc_actor",
            **dataset.provenance.artifact_metadata(),
            **asdict(self.config),
            "final_mse": float(final_loss),
            "actor_architecture": "BoundedActor(18,256,10)",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state_dict": actor.state_dict(),
            "meta": metadata,
        }, output_path)
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return BCTrainingResult(output_path, metadata_path, float(final_loss), metadata)
