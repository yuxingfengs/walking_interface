"""HTWK-style single-stream Actor over the complete 50D velocity-command Observation."""
import numpy as np
import torch
from torch import nn

from .core import DEFAULT_POSE, MAXIMUM, MINIMUM
from .network_inputs import compact, actor_input, actor_input_dim, observation_dim


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers=[]
        self.compact_inputs = compact(config)
        self.observation_dim = observation_dim(config)
        self.input_config = {k:config.get(k) for k in ('network_inputs','actor_observation','command_vy_input')}
        incoming=actor_input_dim(config)
        for width in config['actor_hidden']:
            layers.extend((nn.Linear(incoming,width),nn.ELU()))
            incoming=width
        layers.append(nn.Linear(incoming,12))
        self.network=nn.Sequential(*layers)
        e=config['environment']
        self.register_buffer('low',torch.tensor(np.maximum(
            -1.,(MINIMUM+e['joint_margin']-DEFAULT_POSE)/e['action_scale'])))
        self.register_buffer('high',torch.tensor(np.minimum(
            1.,(MAXIMUM-e['joint_margin']-DEFAULT_POSE)/e['action_scale'])))

    def project(self,action):
        a=action.clamp(-1.,1.)
        hip=.5*(a[...,0:1]+a[...,6:7])
        a=torch.cat((hip,a[...,1:6],hip,a[...,7:]),dim=-1)
        return torch.maximum(torch.minimum(a,self.high),self.low)

    def forward(self,obs):
        if obs.shape[-1] != self.observation_dim:
            raise ValueError(f'Concat Actor expects {self.observation_dim} Observation values')
        return self.project(torch.tanh(self.network(actor_input(obs,self.input_config) if self.compact_inputs else obs)))
