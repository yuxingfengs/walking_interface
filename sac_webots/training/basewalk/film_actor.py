"""Command-FiLM gait with a bounded sensor/yaw correction, one DDPG Actor."""
import numpy as np
import torch
from torch import nn
from .core import DEFAULT_POSE, MINIMUM, MAXIMUM, OBS_DIM


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        widths = config['actor_hidden']
        self.widths = widths
        self.gait_layers = nn.ModuleList()
        incoming = 38  # phase, q, dq, previous Action; no IMU/yaw.
        for width in widths:
            self.gait_layers.append(nn.Linear(incoming, width))
            incoming = width
        self.film = nn.Linear(3, 2*sum(widths))  # Target vx/vy/wz condition every gait layer.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.gait_output = nn.Linear(incoming, 12)
        nn.init.uniform_(self.gait_output.weight, -.003, .003)
        nn.init.zeros_(self.gait_output.bias)
        layers = []
        incoming = 44  # IMU (5), gait state (38), yaw-rate tracking error (1).
        for width in config['correction_hidden']:
            layers.extend([nn.Linear(incoming, width), nn.ELU()])
            incoming = width
        layers.append(nn.Linear(incoming, 12))
        self.correction = nn.Sequential(*layers)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.correction_scale = config['correction_scale']
        e = config['environment']
        self.yaw_rate_scale = e['yaw_rate_scale']
        self.register_buffer('low', torch.tensor(np.maximum(-1., (MINIMUM+e['joint_margin']-DEFAULT_POSE)/e['action_scale'])))
        self.register_buffer('high', torch.tensor(np.minimum(1., (MAXIMUM-e['joint_margin']-DEFAULT_POSE)/e['action_scale'])))

    def project(self, action):
        a = action.clamp(-1., 1.)
        hip = .5*(a[..., 0:1]+a[..., 6:7])
        a = torch.cat((hip, a[..., 1:6], hip, a[..., 7:]), dim=-1)
        return torch.maximum(torch.minimum(a, self.high), self.low)

    def correction_inputs(self, obs):
        # Observation stores scaled actual rate but unscaled command (rad/s).
        rate_error = obs[..., 7:8]*self.yaw_rate_scale - obs[..., 46:47]
        return torch.cat((obs[..., :5], obs[..., 8:46], rate_error), dim=-1)

    def branches(self, obs):
        if obs.shape[-1] != OBS_DIM:
            raise ValueError(f'FiLM Actor expects {OBS_DIM} Observation values')
        conditioning = self.film(obs[..., 5:8])
        h, offset = obs[..., 8:46], 0
        for layer, width in zip(self.gait_layers, self.widths):
            gamma = 1.+conditioning[..., offset:offset+width]
            beta = conditioning[..., offset+width:offset+2*width]
            h = torch.nn.functional.elu(gamma*layer(h)+beta)
            offset += 2*width
        gait = torch.tanh(self.gait_output(h))
        correction = self.correction_scale*torch.tanh(self.correction(self.correction_inputs(obs)))
        return gait, correction

    def forward(self, obs):
        gait, correction = self.branches(obs)
        return self.project(gait+correction)
