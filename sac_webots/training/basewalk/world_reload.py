"""Trusted, same-run continuation at an episode boundary across worldReload."""
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


class WorldReloadRequested(Exception):
    """Unwind/close logs before asking Webots to restart the controller."""


def save_continuation(folder, config, agent, state):
    path = Path(folder)/'world_reload_state.pth'
    temporary = path.with_suffix('.tmp')
    # Includes targets, Adam moments, Replay/cursor/RNG, exploration RNG and BC state.
    payload = dict(version=1, config=config, agent=agent, state=state,
                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate())
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # A Windows file reader can briefly block the atomic rename after fsync.
    deadline = time.monotonic() + 10.0
    while True:
        try:
            temporary.replace(path)
            break
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def load_continuation(folder, config):
    path = Path(folder)/'world_reload_state.pth'
    if not path.exists():
        return None
    # Only our own local run artifact, never an externally supplied checkpoint.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['version'] != 1 or payload['config'] != config:
        raise ValueError('World reload continuation configuration mismatch')
    torch.set_num_threads(1)
    torch.set_rng_state(payload['torch_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    return payload


def reload_world(robot, milliseconds):
    robot.worldReload()
    while robot.step(milliseconds) != -1:
        pass
    # Normal return from the old controller: no simulationQuit/failure notification.
