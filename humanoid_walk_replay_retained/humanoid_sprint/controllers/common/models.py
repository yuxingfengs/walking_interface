"""Neural-network modules shared by training, BC, and Evaluation."""

import torch
import torch.nn as nn

from contracts import ACTION_MAX, ACTION_MIN


class BoundedActor(nn.Module):
    """The existing 18 -> 256 -> 10 Actor with NAO joint-limit clamping."""

    def __init__(self, state_dim: int, hidden_dims, action_dim: int):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

        self.register_buffer("joint_min", torch.tensor(ACTION_MIN, dtype=torch.float32))
        self.register_buffer("joint_max", torch.tensor(ACTION_MAX, dtype=torch.float32))

    def forward(self, state):
        device = self.net[0].weight.device
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return torch.maximum(torch.minimum(self.net(state), self.joint_max), self.joint_min)
