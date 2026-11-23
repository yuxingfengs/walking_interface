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


def motion_reward(command, velocity, gyro_xy, gravity, mode, cfg, *,
                  heading_yaw_rate=None, turn_heading_error=None):
    p = cfg['reward']['biped_tracking']
    values = np.r_[command, velocity, gyro_xy, gravity]
    if not np.isfinite(values).all() or mode not in (0, 1, 2):
        raise ValueError('Invalid biped Reward sensors or source mode')
    error = np.asarray(command) - np.asarray(velocity)
    x = math.exp(-float((error[0]/p['tracking_widths'][0])**2))
    y = math.exp(-float((error[1]/p['tracking_widths'][1])**2))
    xy = x + y
    yaw = p['yaw_weight'] * math.exp(-float((error[2]/p['tracking_widths'][2])**2))
    stability_coefficient = p['turn_stability_coefficient'] if mode == 0 else p['stability_coefficient']
    stability = -stability_coefficient * float(np.asarray(gyro_xy) @ np.asarray(gyro_xy))
    velocity_term = p['velocity_weight'] * (xy+yaw)
    stability_term = p['stability_weight'] * stability
    orientation_raw = float(np.asarray(gravity)[:2] @ np.asarray(gravity)[:2])
    orientation_coefficient = p['turn_orientation_coefficient'] if mode == 0 else p['orientation_coefficient']
    orientation_term = -orientation_coefficient * orientation_raw
    if mode == 0:
        target = float(np.asarray(command)[2])
        if abs(target) <= 1e-6:
            raise ValueError('Turn Reward requires nonzero yaw command')
        width = p['turn_yaw_gaussian_width_rad_s']
        if not math.isfinite(width) or width <= 0:
            raise ValueError('Turn yaw Gaussian width must be positive and finite')
        turning = p['turn_yaw_gaussian_weight'] * math.exp(-((float(velocity[2]) - target) / width)**2)
        xy_penalty_scale = p['turn_xy_penalty_scale']
        x = xy_penalty_scale * (x - 1.)
        y = xy_penalty_scale * (y - 1.)
        xy = x+y
        yaw = turning / p['velocity_weight']
        velocity_term = p['velocity_weight']*(xy+yaw)
    if mode == 2:
        target = float(np.asarray(command)[0])
        if target <= 1e-6:
            raise ValueError('Linear Walk Reward requires positive forward command')
        forward = p['walk_linear_peak'] * (1. - abs(float(np.asarray(velocity)[0])-target)/target)
        velocity_term = forward + p['velocity_weight']*(y-1.) + yaw*p['velocity_weight'] - p['velocity_weight']*p['yaw_weight']
        # Preserve additive decomposition: no hidden constants in reports.
        x = forward/p['velocity_weight']
        y = y-1.
        xy = x+y
        yaw = yaw-p['yaw_weight']
    heading_penalty = 0.0
    if mode == 0 and p.get('turn_heading_weight', 0.):
        if turn_heading_error is None or not math.isfinite(turn_heading_error):
            raise ValueError('Turn requires the causal heading error')
        scale = p['turn_heading_scale_rad']
        distance = (1 - math.cos(turn_heading_error)) / (1 - math.cos(scale))
        heading_penalty = p['turn_heading_weight'] * math.expm1(-distance)
    value = velocity_term + stability_term + orientation_term + heading_penalty
    terms = dict(velocity_tracking=velocity_term, stability=stability_term,
                 orientation=orientation_term, orientation_raw=orientation_raw,
                 turn_heading=heading_penalty,
                 tracking_xy_raw=xy,
                 tracking_yaw_raw=yaw, stability_raw=stability,
                 tracking_x_raw=x, tracking_y_raw=y,
                 tracking_x=p['velocity_weight']*x, tracking_y=p['velocity_weight']*y,
                 tracking_yaw=p['velocity_weight']*yaw,
                 turn_total=value if mode == 0 else 0.,
                 stand_total=value if mode == 1 else 0.,
                 walk_total=value if mode == 2 else 0.,
                 turn_weighted=value if mode != 2 else 0.,
                 walk_weighted=value if mode == 2 else 0.)
    return value, terms


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
