"""Select neural inputs from physical State and optional Actor-only sensors."""
import torch
from torch import nn

REVISION = 'compact_zero_vy_coupled_v1'
PLANAR_COMMAND = 'compact_planar_command_coupled_v1'
OBS_KEEP = tuple(i for i in range(50) if i not in (6,40))
ACTION_KEEP = tuple(i for i in range(12) if i != 6)
ACTOR_NO_LINEAR_VELOCITY = 'without_body_linear_velocity_v1'


ACTOR_NO_LINEAR_OR_SUPPORT = 'without_body_linear_velocity_or_support_v1'
ACTOR_NO_SUPPORT = 'without_support_v1'
ACTOR_BODY_GYRO = 'body_gyro_xyz_without_linear_velocity_or_support_v1'


SHARED_VELOCITY_EMA = 'shared_state_with_velocity_ema_v2'
SHARED_GYRO_EMA = "shared_state_with_gyro_ema_v1"
SHARED_GYRO_RAW = "shared_state_with_raw_gyro_v1"
SHARED_YAW_ONLY = "shared_state_with_yaw_ema_only_v1"
SHARED_STATE_KEEP = OBS_KEEP + (50, 51, 52, 53, 54)


def uses_body_gyro_observation(cfg):
    return cfg.get('actor_observation') in (ACTOR_BODY_GYRO, SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY)


def observation_dim(cfg):
    if cfg.get('actor_observation') in (SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_YAW_ONLY):
        return 55
    return 51 if uses_body_gyro_observation(cfg) else 50


def actor_indices(cfg):
    if cfg.get("actor_observation") == SHARED_YAW_ONLY:
        indices = tuple(i for i in OBS_KEEP if i != 46) + (50, 51)
        return indices + (6,) if cfg.get('network_inputs') == PLANAR_COMMAND else indices
    if cfg.get("actor_observation") == SHARED_GYRO_RAW:
        return tuple(i for i in OBS_KEEP if i != 46) + (50,)
    if cfg.get("actor_observation") == SHARED_GYRO_EMA:
        return tuple(i for i in SHARED_STATE_KEEP if i not in (46, 52))
    keep = OBS_KEEP if compact(cfg) else tuple(range(50))
    if cfg.get('actor_observation') == SHARED_VELOCITY_EMA:
        return SHARED_STATE_KEEP
    if cfg.get('actor_observation') == ACTOR_BODY_GYRO:
        return tuple(i for i in keep if i < 46) + (50,)
    if cfg.get('actor_observation') in (ACTOR_NO_LINEAR_VELOCITY, ACTOR_NO_LINEAR_OR_SUPPORT):
        keep = tuple(i for i in keep if i not in (47,48))
    if cfg.get('actor_observation') in (ACTOR_NO_LINEAR_OR_SUPPORT, ACTOR_NO_SUPPORT):
        keep = tuple(i for i in keep if i != 49)
    return keep


def actor_input_dim(cfg):
    return len(actor_indices(cfg))


def compact(cfg):
    return cfg.get('network_inputs') == REVISION


def critic_keep(extra):
    return OBS_KEEP + tuple(range(50,50+extra)) + tuple(50+extra+i for i in ACTION_KEEP)


def actor_input(obs, cfg=None):
    if (cfg or {}).get('network_inputs') != PLANAR_COMMAND and torch.any(obs[...,6].abs() > 1e-8):
        raise ValueError('Compact inputs require command_vy=0')
    if not torch.allclose(obs[...,34],obs[...,40],rtol=0,atol=1e-7):
        raise ValueError('Compact inputs require tied previous HipYawPitch Actions')
    return obs[...,list(actor_indices(cfg))] if cfg is not None else obs[...,list(OBS_KEEP)]


class ProjectedCritic(nn.Sequential):
    """Accept the physical packed interface, feed only unique neural columns."""
    def __init__(self, modules, extra, obs_dim=50, shared_state=None):
        super().__init__(*modules)
        self.start = obs_dim + extra
        self.planar_command = bool(shared_state and 6 in shared_state)
        if shared_state:
            self.keep = shared_state + tuple(range(obs_dim, self.start)) + tuple(
                self.start + i for i in ACTION_KEEP)
        else:
            self.keep = tuple(i if i < 50 else i + obs_dim - 50 for i in critic_keep(extra))

    def forward(self, packed):
        if self.planar_command:
            if not torch.allclose(packed[...,34], packed[...,40], rtol=0, atol=1e-7):
                raise ValueError('Expected tied previous HipYawPitch actions')
        else:
            actor_input(packed[...,:50])
        start = self.start
        if not torch.allclose(packed[...,start],packed[...,start+6],rtol=0,atol=1e-7):
            raise ValueError('Compact Critic requires physically tied executed Actions')
        return super().forward(packed[...,list(self.keep)])


def critic_input_dim(cfg,extra):
    if cfg.get('actor_observation') in (SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY):
        return len(actor_indices(cfg)) + extra + len(ACTION_KEEP)
    return 48+extra+11 if compact(cfg) else 50+extra+12
