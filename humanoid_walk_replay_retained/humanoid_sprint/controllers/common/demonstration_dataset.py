"""Validated, contract-aware demonstration datasets for Behavior Cloning."""

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

import numpy as np
import torch

from contracts import (
    ACTION_DIM,
    STATE_DIM,
    contract_metadata,
    validate_contract_metadata,
)


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


@dataclass(frozen=True)
class DemonstrationProvenance:
    dataset_path: str
    dataset_sha256: str
    samples: int
    expert_id: str
    uses_privileged_state: Optional[bool]
    is_legacy: bool
    sidecar_path: Optional[str]

    def artifact_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "samples": self.samples,
            "expert_id": self.expert_id,
            "uses_privileged_state": self.uses_privileged_state,
            "source_is_legacy": self.is_legacy,
        }


@dataclass(frozen=True)
class DemonstrationDataset:
    states: torch.Tensor
    actions: torch.Tensor
    provenance: DemonstrationProvenance

    @classmethod
    def load(cls, path: Path) -> "DemonstrationDataset":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"expert dataset not found: {path}")

        required_fields = (
            [f"state_{index}" for index in range(STATE_DIM)]
            + [f"action_{index}" for index in range(ACTION_DIM)]
        )
        states, actions = [], []
        first_row = None
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_fields = [
                field for field in required_fields if field not in (reader.fieldnames or [])
            ]
            if missing_fields:
                raise ValueError(
                    f"demonstration schema missing fields: {', '.join(missing_fields)}"
                )
            for row_index, row in enumerate(reader):
                if first_row is None:
                    first_row = row
                state = [float(row[f"state_{index}"]) for index in range(STATE_DIM)]
                action = [float(row[f"action_{index}"]) for index in range(ACTION_DIM)]
                if not all(math.isfinite(value) for value in state + action):
                    raise ValueError(f"non-finite demonstration value at row {row_index + 2}")
                states.append(state)
                actions.append(action)

        if not states or first_row is None:
            raise ValueError(f"expert dataset is empty: {path}")

        row_contracts = {key: first_row.get(key) for key in contract_metadata()}
        is_versioned = validate_contract_metadata(row_contracts, artifact=str(path))
        expert_id = str(first_row.get("expert_id") or "legacy_unknown")
        privileged = _parse_bool(first_row.get("uses_privileged_state"))

        sidecar_path = path.with_suffix(".metadata.json")
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            validate_contract_metadata(sidecar, artifact=str(sidecar_path))
            for key, row_value in row_contracts.items():
                if row_value and sidecar.get(key) != row_value:
                    raise ValueError(
                        f"dataset/sidecar contract mismatch for {key}: "
                        f"{row_value!r} != {sidecar.get(key)!r}"
                    )
            if first_row.get("expert_id") and sidecar.get("expert_id") != expert_id:
                raise ValueError("dataset/sidecar expert_id mismatch")
            sidecar_privileged = _parse_bool(sidecar.get("uses_privileged_state"))
            if privileged is not None and sidecar_privileged != privileged:
                raise ValueError("dataset/sidecar privileged-state declaration mismatch")
            privileged = sidecar_privileged
        elif is_versioned:
            warnings.warn(
                f"versioned demonstration has no sidecar metadata: {path}",
                RuntimeWarning,
                stacklevel=2,
            )

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            states=torch.tensor(np.asarray(states), dtype=torch.float32),
            actions=torch.tensor(np.asarray(actions), dtype=torch.float32),
            provenance=DemonstrationProvenance(
                dataset_path=str(path.resolve()),
                dataset_sha256=digest,
                samples=len(states),
                expert_id=expert_id,
                uses_privileged_state=privileged,
                is_legacy=not is_versioned,
                sidecar_path=str(sidecar_path.resolve()) if sidecar_path.is_file() else None,
            ),
        )
