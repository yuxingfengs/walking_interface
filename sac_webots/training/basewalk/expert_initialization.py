"""Reuse baseline's expert initialization, never its trained networks."""
import copy
import torch
from .expert_pipeline import initialize, signature
from .core import critic_extra_dim


def initialize_expert(agent, cfg):
    saved = torch.load(cfg['expert_pipeline']['dataset'], map_location='cpu', weights_only=False)
    source = copy.deepcopy(cfg)
    source.pop('expert_initialization')
    # Dataset and pretraining retain their original angle coverage. Only the
    # online angle list may differ; all reward/state/task checks remain strict.
    source['direction_training']['angles_deg'] = saved['signature']['direction']['angles_deg']
    if signature(source) != saved['signature']:
        raise ValueError('Expert initialization contract differs beyond online angles')
    initialize(agent, source)
    agent.initial_actor_source = dict(kind='expert_bc',
        checkpoint=cfg['initialization']['bc_checkpoint'], sha256=cfg['initialization']['bc_sha256'])
