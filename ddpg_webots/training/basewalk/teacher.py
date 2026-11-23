"""Read-only 46D MLP teacher copied into this standalone project."""
from pathlib import Path
import hashlib
import torch
from torch import nn

from . import TRAINING_ROOT as ROOT
TEACHER = ROOT/'bc_candidates/sin_mixed_46d_v1/actor.pth'
TEACHER_SHA256 = '87b9cba38b5162316b310a8c2ae1f8e4ff0f73965952f0584123abf0945c4f46'


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        if hashlib.sha256(TEACHER.read_bytes()).hexdigest() != TEACHER_SHA256:
            raise ValueError('Teacher checkpoint hash mismatch')
        payload = torch.load(TEACHER, map_location='cpu', weights_only=False)
        if payload['contract']['version'] != 'nao_basewalk_46d_12d_coupled_v2' or payload['config'].get('actor_type', 'mlp') != 'mlp':
            raise ValueError('Only the archived 46D MLP teacher is supported')
        self.config = payload['config']
        layers, size = [], 46
        for width in self.config['actor_hidden']:
            layers.extend([nn.Linear(size, width), nn.ELU()]);size = width
        layers.append(nn.Linear(size, 12))
        self.net = nn.Sequential(*layers)
        self.register_buffer('low', torch.empty(12))
        self.register_buffer('high', torch.empty(12))
        self.load_state_dict(payload['actor'], strict=True)
        self.eval().requires_grad_(False)

    def forward(self, obs):
        a = torch.tanh(self.net(obs))
        hip = .5*(a[..., :1]+a[..., 6:7])
        a = torch.cat((hip, a[..., 1:6], hip, a[..., 7:]), -1)
        return torch.maximum(torch.minimum(a, self.high), self.low)
