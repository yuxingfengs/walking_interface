"""Trusted, same-run continuation at an episode boundary across worldReload."""
import os
import random
from pathlib import Path

import numpy as np
import torch


class WorldReloadRequested(Exception):
    """Unwind/close logs before asking Webots to restart the controller."""


def save_continuation(folder, config, agent, state, *, filename='world_reload_state.pth'):
    path = Path(folder)/filename
    temporary = path.with_suffix('.tmp')
    # Includes targets, Adam moments, Replay/cursor/RNG, exploration RNG and BC state.
    payload = dict(version=1, config=config, agent=agent, state=state,
                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate())
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_continuation(folder, config, *, checkpoint=None):
    path = (Path(folder)/'world_reload_state.pth' if checkpoint is None else
            Path(checkpoint['path']).with_name(f"continuation_{checkpoint['step']:08d}.pth"))
    if not path.exists():
        return None
    # Only our own local run artifact, never an externally supplied checkpoint.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint is not None:
        if (payload['state'].get('checkpoint_sha256') != checkpoint['sha256']
                or payload['state']['next_step'] != checkpoint['step'] + 1):
            raise ValueError('Final continuation does not match the selected step checkpoint')
        from .run_storage import group_name
        old = payload['config']
        training = dict(old['training'], total_env_steps=config['training']['total_env_steps'],
                        checkpoint_start=checkpoint)
        if group_name(dict(old, training=training)) != group_name(config):
            raise ValueError('Final continuation requires the same training configuration')
        payload['config'] = config
        payload['agent'].config.update(training=config['training'])
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
