"""Input helpers shared by training and Behavior Cloning."""

import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def load_initial_actions_csv(
    filename: Union[str, Path],
    expected_dim: Optional[int] = None,
    to_radians: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    rows = []
    with Path(filename).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([float(value) for value in row])
    actions = np.asarray(rows, dtype=np.float32)
    if expected_dim is not None and actions.shape[1] != expected_dim:
        raise ValueError(f"initial_actions dim mismatch: got {actions.shape[1]}, expect {expected_dim}")
    if to_radians:
        actions = np.deg2rad(actions)
    return [torch.tensor(action, dtype=dtype, device=device) for action in actions]
