"""DDPG ablation: remove only Critic normalization, preserving action placement."""
from torch import nn

from algorithms.ddpg import Critic, DDPGAlgorithm


class NoNormCritic(Critic):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()


class DDPGNoNormAlgorithm(DDPGAlgorithm):
    algorithm_id = 'ddpg_no_norm_v1'

    def __init__(self, **kwargs):
        if kwargs.get('sync_bn_running_stats', False):
            raise ValueError('No-normalization Critic requires sync_bn_running_stats=False')
        super().__init__(**kwargs)

    def create_critic(self, state_dim, action_dim, hidden_dim):
        return NoNormCritic(state_dim, action_dim, hidden_dim)
