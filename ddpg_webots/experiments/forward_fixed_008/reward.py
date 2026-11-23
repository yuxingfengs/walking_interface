"""Walk with a commanded torso-frame planar velocity and zero requested yaw rate."""
import math
import numpy as np


def motion_reward(command, velocity, gyro_xy, gravity, mode, cfg, **_):
    if mode != 2 or command[2] != 0 or not np.isfinite(np.r_[command, velocity, gyro_xy, gravity]).all():
        raise ValueError('Expected a finite Walk transition with zero commanded yaw rate')
    p = cfg['reward']['biped_tracking']
    error = (np.asarray(velocity[:2]) - command[:2]) / p['walk_tracking_widths_m_s']
    scores = np.exp(-np.square(error))
    peak = p['walk_linear_peak']
    planar = peak * scores.prod()
    yaw = p['velocity_weight'] * p['yaw_weight'] * math.expm1(-(velocity[2]/p['tracking_widths'][2])**2)
    stability = -p['stability_weight'] * p['stability_coefficient'] * np.dot(gyro_xy, gyro_xy)
    tilt = -p['orientation_coefficient'] * np.dot(gravity[:2], gravity[:2])
    total = float(planar + yaw + stability + tilt)
    # The two axis diagnostics are individual scores; their product determines planar reward.
    return total, dict(tracking_x=float(peak*scores[0]), tracking_y=float(peak*scores[1]),
                       tracking_xy=float(planar), tracking_yaw=float(yaw), velocity_tracking=float(planar+yaw),
                       stability=float(stability), orientation=float(tilt), walk_weighted=total, turn_weighted=0.)


def observation_reward(obs, nxt, mode, cfg, *, reward_velocity=None, **_):
    if reward_velocity is None:
        reward_velocity = [nxt[47], nxt[48], nxt[50]/cfg['environment']['yaw_rate_scale']]
    return motion_reward(obs[5:8], reward_velocity, nxt[3:5], nxt[:3], mode, cfg)


def filtered_reward_velocities(arrays, cfg):
    nxt = np.asarray(arrays.get('next_observations', arrays.get('next_obs')))
    velocities = np.column_stack((nxt[:, 47:49], nxt[:, 50]/cfg['environment']['yaw_rate_scale']))
    if 'reward_velocities' in arrays and not np.allclose(arrays['reward_velocities'], velocities, atol=1e-7):
        raise ValueError('Saved Walk velocities disagree with next-state velocity and raw body gyro')
    return velocities


def revalue_arrays(arrays, cfg):
    obs = arrays.get('observations', arrays.get('obs'))
    nxt = arrays.get('next_observations', arrays.get('next_obs'))
    velocities = filtered_reward_velocities(arrays, cfg)
    if not len(obs)==len(nxt)==len(arrays['modes'])==len(velocities):
        raise ValueError('Unaligned Walk arrays')
    return np.array([observation_reward(o, n, int(m), cfg, reward_velocity=v)[0]
                     for o, n, m, v in zip(obs, nxt, np.ravel(arrays['modes']), velocities)])
