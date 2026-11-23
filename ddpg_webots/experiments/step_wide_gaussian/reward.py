"""Turn scores raw body gyro Z with a fixed-width Gaussian.

Score explicit post-action Reward velocities against the source command.
Task-reset EMA or current velocities are supplied by the environment or ordered replay;
Actor Observation remains instantaneous and filtering is never applied twice.
Units: m/s, rad/s. No dt multiplier or affine calibration.
Walk/Stand and Turn translation continue to use the configured velocity source.
"""
import math
import numpy as np
from basewalk.core import OBS_DIM


def motion_reward(command, velocity, gyro_xy, gravity, mode, cfg, **unused):
    p = cfg['reward']['biped_tracking']
    values = np.r_[command, velocity, gyro_xy, gravity]
    if not np.isfinite(values).all() or mode not in (0, 1, 2):
        raise ValueError('Invalid tracking inputs')
    error = np.asarray(velocity) - np.asarray(command)
    x = p['turn_walk_xy_peak'] * math.exp(-float((error[0]/p['turn_walk_xy_width_m_s'])**2))
    y = p['turn_walk_xy_peak'] * math.exp(-float((error[1]/p['turn_walk_xy_width_m_s'])**2))
    yaw = p['turn_yaw_gaussian_weight'] * math.exp(-float((error[2]/p['turn_yaw_gaussian_width_rad_s'])**2))
    value = x + y + yaw
    return value, dict(velocity_tracking=value, stability=0., orientation=0.,
        orientation_raw=float(np.asarray(gravity)[:2] @ np.asarray(gravity)[:2]),
        turn_heading=0., tracking_xy_raw=x+y, tracking_yaw_raw=yaw,
        stability_raw=0., tracking_x_raw=x, tracking_y_raw=y,
        tracking_x=x, tracking_y=y, tracking_yaw=yaw,
        turn_total=value if mode==0 else 0., stand_total=value if mode==1 else 0.,
        walk_total=value if mode==2 else 0., turn_weighted=value if mode!=2 else 0.,
        walk_weighted=value if mode==2 else 0.)


def observation_reward(obs, nxt, mode, cfg, *, reward_velocity=None, turn_heading_error=None):
    obs, nxt = np.asarray(obs), np.asarray(nxt)
    if obs.shape != nxt.shape or obs.shape not in ((51,), (55,)):
        raise ValueError('Reward requires aligned physical observations with gyro Z')
    if reward_velocity is None:
        raise ValueError('Yaw EMA Reward requires ordered history or explicit Reward velocity')
    velocity = np.asarray(reward_velocity)
    if velocity.shape != (3,):
        raise ValueError('Reward requires 3D velocity')
    return motion_reward(obs[5:8], velocity, nxt[3:5], nxt[:3], int(mode), cfg, turn_heading_error=turn_heading_error)


def filtered_reward_velocities(arrays, cfg):
    from basewalk.reward_velocity import sequence_velocities
    physical = {}
    for key in ('observations', 'next_observations', 'obs', 'next_obs'):
        if key in arrays:
            source = np.asarray(arrays[key])
            values = source[:, :OBS_DIM].copy()
            values[:, 46] = source[:, 50]
            physical[key] = values
    scoring = dict(arrays, **physical)
    result = sequence_velocities(scoring, cfg)
    if 'reward_velocities' in arrays and not np.allclose(result,np.asarray(arrays['reward_velocities']),rtol=1e-5,atol=1e-7):
        raise ValueError('Saved Reward velocities do not match the selected source')
    return result


def step_rewards(arrays, cfg):
    obs = arrays.get('observations', arrays.get('obs'))
    nxt = arrays.get('next_observations', arrays.get('next_obs'))
    modes = arrays['modes']
    if obs is None or nxt is None or len(obs) != len(nxt) or len(obs) != len(modes):
        raise ValueError('Missing aligned source/next observations for Reward export')
    errors = (np.asarray(arrays['turn_heading_error']).reshape(-1)
              if cfg['reward']['biped_tracking'].get('turn_heading_weight', 0.)
              else np.zeros(len(modes)))
    if len(errors) != len(modes):
        raise ValueError('Heading errors must match transition count')
    velocities = filtered_reward_velocities(arrays, cfg)
    return np.array([observation_reward(o, n, m, cfg, reward_velocity=v, turn_heading_error=e)[0]
                     for o, n, m, v, e in zip(obs, nxt, modes, velocities, errors)])


def revalue_arrays(arrays, cfg):
    return step_rewards(arrays, cfg)
