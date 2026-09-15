"""Frozen BC plus a bounded, trainable residual in absolute joint-action space."""
import torch
from torch import nn
from models import BoundedActor


class ResidualActor(BoundedActor):
    def __init__(self, state_dim=25, hidden_dim=256, action_dim=10, residual_limit_rad=.03):
        if state_dim != 25 or not 0 < residual_limit_rad <= .1:
            raise ValueError('Residual heading Actor requires 25D State and limit in (0,.1]')
        super().__init__(state_dim,hidden_dim,action_dim)
        self.base = BoundedActor(24,hidden_dim,action_dim)
        self.base.requires_grad_(False)
        self.register_buffer('residual_limit',torch.tensor(float(residual_limit_rad)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def state_tensor(self, state):
        state=torch.as_tensor(state,dtype=torch.float32,device=self.joint_min.device)
        return state.unsqueeze(0) if state.dim()==1 else state

    def base_action(self, state):
        return self.base(self.state_tensor(state)[...,:24])

    def forward(self, state):
        state=self.state_tensor(state)
        action=self.base_action(state)+self.residual_limit*torch.tanh(self.net(state))
        return torch.maximum(torch.minimum(action,self.joint_max),self.joint_min)

    def limit_action(self, state, action):
        base=self.base_action(state)
        action=torch.as_tensor(action,device=base.device,dtype=base.dtype)
        action=torch.maximum(torch.minimum(action,base+self.residual_limit),base-self.residual_limit)
        return torch.maximum(torch.minimum(action,self.joint_max),self.joint_min)

    def load_state_dict(self, state_dict, strict=True):
        saved=state_dict.get('residual_limit')
        if saved is None or abs(float(saved)-float(self.residual_limit)) > 1e-7:
            raise ValueError('Residual checkpoint correction limit does not match configuration')
        return super().load_state_dict(state_dict,strict=strict)
