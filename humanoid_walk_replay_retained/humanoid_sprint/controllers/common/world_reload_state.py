"""Private, same-run continuation across a Webots world reload (not a model resume)."""
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_continuation(snapshot, algorithm, metadata):
    snapshot = Path(snapshot)
    run_id = json.loads(snapshot.read_text(encoding='utf-8'))['run_id']
    state = snapshot.parent / 'world_reload_state.pth'
    temporary = state.with_suffix('.tmp')
    # Pickle preserves optimizer/parameter references, all target networks and Replay.
    # This file is internal to this run; never accept arbitrary downloaded objects.
    torch.save(dict(run_id=run_id, algorithm=algorithm, metadata=metadata, rng=capture_rng()), str(temporary))
    os.replace(str(temporary), str(state))
    marker = snapshot.parent / 'world_reload_pending.json'
    marker.write_text(json.dumps(dict(pending=True, run_id=run_id)), encoding='utf-8')


def load_continuation(snapshot):
    snapshot = Path(snapshot)
    marker = snapshot.parent / 'world_reload_pending.json'
    if not marker.exists():
        return None
    request = json.loads(marker.read_text(encoding='utf-8'))
    if not request['pending']:
        return None
    run_id = json.loads(snapshot.read_text(encoding='utf-8'))['run_id']
    if request['run_id'] != run_id:
        raise ValueError('World reload continuation belongs to another run')
    bundle = torch.load(str(snapshot.parent / 'world_reload_state.pth'), weights_only=False)
    if bundle['run_id'] != run_id:
        raise ValueError('World reload state belongs to another run')
    marker.write_text(json.dumps(dict(pending=False, run_id=run_id)), encoding='utf-8')
    return bundle
