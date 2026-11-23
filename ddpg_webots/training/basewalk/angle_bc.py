"""Promote the last qualifying training Actor at a completed angle boundary."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from .core import CONTRACT


def bc_source(agent):
    return getattr(agent, 'bc_source', dict(kind='initial_expert' if agent.bc_state is not None else 'none',
        checkpoint=agent.config.get('initialization', {}).get('bc_checkpoint')))


def remember_success(agent, actor_state, folder, metadata):
    # Snapshot is the policy that produced the final Action, before its update.
    # An online learning episode is not an independent fixed-policy evaluation.
    path = Path(folder)/'successful_actors'/f"angle_{metadata['angle_deg']:g}_step_{metadata['step']:08d}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = dict(kind='training_success', checkpoint=str(path.relative_to(folder)), **metadata)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(contract=CONTRACT, config=agent.config, actor=actor_state,
        env_steps=metadata['step'], source=source,
        checkpoint_kind='training_success_actor_only', independent_evaluation='pending'), temporary)
    temporary.replace(path)
    source['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    agent.angle_bc_candidate = dict(actor=actor_state, source=source)


def finish_angle(agent, folder, angle, step):
    candidate = getattr(agent, 'angle_bc_candidate', None)
    if candidate is not None:
        if candidate['source']['angle_deg'] != angle:
            raise ValueError('Successful Actor belongs to a different angle')
        agent.bc_state = copy.deepcopy(candidate['actor'])
        agent.bc_source = copy.deepcopy(candidate['source'])
    if agent.bc_state is not None:
        # Existing reset semantics: Actor/target + fresh Actor Adam; Critic/Replay retained.
        agent.restore_bc()
    agent.angle_bc_candidate = None
    record = dict(step=step, completed_angle_deg=angle, promoted=candidate is not None,
                  fallback=candidate is None, source=bc_source(agent), replay_count=agent.replay.count)
    with (Path(folder)/'bc_handoffs.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, allow_nan=False)+'\n')
    return record
