"""Read-only Q of a source state and the action actually executed."""
import numpy as np
import torch

def executed_action_q(critic,obs,action,features=None):
    parts=[np.asarray(obs,dtype=np.float32).reshape(-1)]
    if features is not None:parts.append(np.asarray(features,dtype=np.float32).reshape(-1))
    parts.append(np.asarray(action,dtype=np.float32).reshape(-1))
    parameter=next(critic.parameters())
    x=torch.as_tensor(np.concatenate(parts),device=parameter.device,dtype=parameter.dtype).unsqueeze(0)
    with torch.no_grad():value=critic(x).reshape(-1)
    if value.numel()!=1 or not torch.isfinite(value).all():raise ValueError('Invalid executed-action Q')
    return float(value.item())
