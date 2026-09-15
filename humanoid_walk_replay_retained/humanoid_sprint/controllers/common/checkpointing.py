"""Checkpoint helpers that keep Evaluation compatible with training output."""

from pathlib import Path
import json
import os
from typing import Mapping, Optional, Union

import torch

from contracts import contract_metadata, validate_contract_metadata


def load_torch_artifact(filepath: Union[str, Path], map_location):
    """Load a trusted project artifact without PyTorch's implicit-mode warning."""
    try:
        return torch.load(Path(filepath), map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that predate weights_only.
        return torch.load(Path(filepath), map_location=map_location)


def save_ddpg_checkpoint(host, filepath: Union[str, Path], meta: Optional[Mapping] = None) -> Path:
    """Save the current DDPG state using the project's existing key schema."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "actor_state_dict": host.actor.state_dict(),
        "critic_state_dict": host.critic.state_dict(),
        "actor_target_state_dict": host.actor_target.state_dict(),
        "critic_target_state_dict": host.critic_target.state_dict(),
        "actor_optimizer": host.actor_optimizer.state_dict(),
        "critic_optimizer": host.critic_optimizer.state_dict(),
        "train_logs": getattr(host, "_train_logs", None),
    }
    checkpoint["meta"] = {
        **dict(meta or {}),
        **contract_metadata(task_id=getattr(host, "task_id", "forward_walk_v1"),
                            state_dim=getattr(host, 'state_dim', 18)),
        "algorithm_id": getattr(host, "algorithm_id", "ddpg_v1"),
    }
    torch.save(checkpoint, path)
    snapshot = os.environ.get("WALKING_RUN_CONFIG")
    if snapshot:
        payload = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        if hasattr(host, 'actor_update_count'):
            payload['checkpoint_training_state'] = {
                'actor_update_count': host.actor_update_count,
                'critic_warmup_completed': host.critic_warmup_completed,
            }
        path.with_suffix(".config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return path


def load_actor_weights(actor, filepath: Union[str, Path], device: torch.device, *, expected_task_id: str = "forward_walk_v1") -> None:
    """Load actor weights from both current and legacy checkpoint schemas."""
    checkpoint = load_torch_artifact(filepath, device)
    if "actor_state_dict" in checkpoint:
        validate_contract_metadata(checkpoint.get("meta"), artifact=str(filepath), expected_task_id=expected_task_id, expected_state_dim=actor.net[0].in_features)
        actor.load_state_dict(checkpoint["actor_state_dict"])
    elif "actor" in checkpoint:
        validate_contract_metadata(checkpoint.get("meta"), artifact=str(filepath), expected_task_id=expected_task_id, expected_state_dim=actor.net[0].in_features)
        actor.load_state_dict(checkpoint["actor"])
    else:
        # Legacy BC artifacts were saved as a bare state_dict.
        validate_contract_metadata(None, artifact=str(filepath), expected_task_id=expected_task_id, expected_state_dim=actor.net[0].in_features)
        actor.load_state_dict(checkpoint)
    actor.to(device)
    actor.eval()
