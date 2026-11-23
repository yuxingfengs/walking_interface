"""Dispatch scoring/revaluation to the selected experiment's verified Reward."""
import hashlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
from .core import OBS_DIM
from .network_inputs import uses_body_gyro_observation

_CACHE = {}


def implementation(cfg):
    if cfg.get('reward_velocity_source_sha256'):
        path=Path(__file__).with_name('reward_velocity.py')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['reward_velocity_source_sha256']:
            raise ValueError('Reward velocity source changed; regenerate assets and pretrain Critic')
    if cfg.get('reward', {}).get('biped_tracking', {}).get('velocity_source') == 'causal_mean':
        history_path = Path(__file__).with_name('reward_history.py')
        if hashlib.sha256(history_path.read_bytes()).hexdigest() != cfg.get('reward_history_sha256'):
            raise ValueError('Reward history implementation changed; derive compatible assets and pretrain Critic')
    path = Path(cfg['reward_module_path'])
    expected = cfg['reward_source_sha256']
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Reward source changed after configuration/snapshot; refusing mixed-version training')
    key = (str(path.resolve()), actual)
    if key not in _CACHE:
        name = 'experiment_reward_'+actual
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        for attr in ('motion_reward', 'observation_reward', 'filtered_reward_velocities', 'revalue_arrays'):
            if not callable(getattr(module, attr, None)):
                raise ValueError('Reward missing required interface: '+attr)
        _CACHE[key] = module
    return _CACHE[key]


def motion_reward(command, velocity, gyro_xy, gravity, mode, cfg, *,
                  actual_yaw_rate=None, previous_gravity=None, body_gyro_z=None,
                  action=None, previous_action=None, joint_position=None, reverse_foot_yaw_cost=None,
                  turn_heading_error=None, previous_body_gyro_z=None, reward_dt=None, gait_phase=None):
    score = implementation(cfg).motion_reward
    params = cfg['reward'].get('biped_tracking', {})
    sensors = {}
    if mode == 0 and params.get('phase_foot'):
        sensors.update(gait_phase=gait_phase, joint_position=joint_position)
    if 'turn_yaw_overspeed_max_penalty' in params:
        sensors['actual_yaw_rate'] = body_gyro_z if uses_body_gyro_reward(cfg) else actual_yaw_rate
    if uses_heading_yaw_tracking(cfg):
        sensors['heading_yaw_rate'] = actual_yaw_rate
    if 'turn_tilt_divergence_max_penalty' in params:
        sensors['previous_gravity'] = previous_gravity
    if mode == 0 and params.get('turn_world_z_regularization'):
        sensors.update(body_gyro_z=body_gyro_z, action=action,
                       previous_action=previous_action, joint_position=joint_position)
    if mode == 0 and params.get('turn_reverse_foot_yaw_weight'):
        sensors['reverse_foot_yaw_cost'] = reverse_foot_yaw_cost
    if mode == 0 and params.get('turn_heading_window_seconds'):
        sensors['turn_heading_error'] = turn_heading_error
    if mode == 0 and (params.get('turn_yaw_error_progress_weight') or params.get('turn_yaw_potential_weight') or params.get('turn_acceleration_weight')):
        sensors['previous_body_gyro_z'] = previous_body_gyro_z
    if mode == 0 and params.get('turn_acceleration_weight'):
        sensors['reward_dt'] = reward_dt
    return score(command, velocity, gyro_xy, gravity, mode, cfg, **sensors)


def uses_body_gyro_reward(cfg):
    return cfg['reward'].get('biped_tracking', {}).get('turn_yaw_sensor') == 'body_gyro_z'


def uses_heading_yaw_tracking(cfg):
    return cfg['reward'].get('biped_tracking', {}).get('turn_yaw_tracking_sensor') == 'heading_delta'


def retains_body_gyro(cfg):
    params = cfg['reward']['biped_tracking']
    return params.get('yaw_ema_use_body_gyro') or params.get('turn_world_z_regularization')


def observation_reward(obs, nxt, mode, cfg, *, reward_velocity=None):
    if uses_body_gyro_observation(cfg) and not uses_body_gyro_reward(cfg) and not retains_body_gyro(cfg):
        obs, nxt = obs[:OBS_DIM], nxt[:OBS_DIM]
    return implementation(cfg).observation_reward(obs, nxt, mode, cfg, reward_velocity=reward_velocity)


def reward_arrays(arrays, cfg):
    if not uses_body_gyro_observation(cfg) or uses_body_gyro_reward(cfg) or retains_body_gyro(cfg):
        return arrays
    return {key: np.asarray(value)[..., :OBS_DIM]
            if key in ('observations', 'next_observations', 'obs', 'next_obs') else value
            for key, value in arrays.items()}


def filtered_reward_velocities(arrays, cfg):
    return implementation(cfg).filtered_reward_velocities(reward_arrays(arrays, cfg), cfg)


def revalue_arrays(arrays, cfg):
    return implementation(cfg).revalue_arrays(reward_arrays(arrays, cfg), cfg)
