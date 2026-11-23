"""Causal cycle-scale translation estimate and tolerance-based turn reward."""
from collections import deque
import math
import numpy as np


class NetSpeedWindow:
    def __init__(self, samples, dt):
        self.positions = deque(maxlen=samples)
        self.samples, self.dt = samples, dt
        self.speed, self.fill = 0., 0.

    @property
    def ready(self):
        return len(self.positions) == self.samples

    def advance(self, position):
        point = np.asarray(position, dtype=float)[:2]
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Net speed requires finite planar position')
        self.positions.append(point.copy())
        n = len(self.positions)
        self.fill = n / self.samples
        self.speed = (float(np.linalg.norm(self.positions[-1]-self.positions[0])) /
                      ((n-1)*self.dt)) if n > 1 else 0.
        return self.speed


class PerStepPlanarSpeed:
    """World-horizontal displacement over one control step; no rolling window."""
    def __init__(self, dt, position):
        self.dt = float(dt)
        self.previous = np.asarray(position, dtype=float)[:2].copy()
        self.speed, self.fill, self.ready = 0., 0., False

    def advance(self, position):
        point = np.asarray(position, dtype=float)[:2]
        if not np.isfinite(point).all() or self.dt <= 0:
            raise ValueError('Invalid per-step planar speed input')
        self.speed = float(np.linalg.norm(point-self.previous)) / self.dt
        self.previous = point.copy()
        self.fill, self.ready = 1., True
        return self.speed


def stationary_terms(old_error, new_error, yaw_error, speed, ready, cfg, dt, fell=False):
    """Errors in radians; progress gain is explicitly per degree, not per second."""
    p, w = cfg['stationary_reward'], cfg['reward']
    values = (old_error, new_error, yaw_error, speed, dt)
    if not all(math.isfinite(x) for x in values) or speed < 0 or dt <= 0:
        raise ValueError('Invalid stationary reward inputs')
    old = max(abs(math.degrees(old_error))-p['angle_tolerance_deg'], 0.)
    new = max(abs(math.degrees(new_error))-p['angle_tolerance_deg'], 0.)
    excess_speed = max(speed-p['speed_tolerance_m_s'], 0.)
    excess_yaw = max(abs(yaw_error)-p['yaw_tolerance_rad_s'], 0.)
    hold = ready and new == 0. and excess_speed == 0. and excess_yaw == 0. and not fell
    progress = float(np.clip(p['progress_per_degree']*(old-new),
                             -p['progress_limit'], p['progress_limit']))
    if hold:
        progress = 0.
    elif progress > 0 and (not ready or fell):
        progress = 0.
    return dict(turn_hold=p['hold_bonus'] if hold else 0.,
        turn_heading_progress=progress,
        turn_heading_error=-w['turn_heading_error']*(math.radians(new)/math.pi)**2*dt,
        turn_linear_velocity=-w['turn_linear_velocity_penalty']*speed*dt,
        turn_yaw_rate_error=-w['turn_yaw_rate_error']*excess_yaw**2*dt,
        turn_translation=0., walk_forward_progress=0., walk_heading_progress=0.,
        walk_heading_error=0., walk_cross_track=0.)


def stand_terms(yaw_rate, speed, ready, cfg, dt, fell=False):
    """Zero-command stability reward; independent of task heading and progress."""
    p, w = cfg['stationary_reward'], cfg['reward']
    if not all(math.isfinite(x) for x in (yaw_rate, speed, dt)) or speed < 0 or dt <= 0:
        raise ValueError('Invalid Stand reward inputs')
    excess_yaw = max(abs(yaw_rate)-p['yaw_tolerance_rad_s'], 0.)
    hold = ready and speed <= p['speed_tolerance_m_s'] and excess_yaw == 0. and not fell
    # Existing term names retained for CSV/replay compatibility; stand_total labels the stage.
    return dict(turn_hold=p['hold_bonus'] if hold else 0.,
        turn_heading_progress=0., turn_heading_error=0.,
        turn_linear_velocity=-w.get('stand_linear_velocity_penalty',w['turn_linear_velocity_penalty'])*speed*dt,
        turn_yaw_rate_error=-w['turn_yaw_rate_error']*excess_yaw**2*dt,
        turn_translation=0., walk_forward_progress=0., walk_heading_progress=0.,
        walk_heading_error=0., walk_cross_track=0.)


def critic_features(env, cfg):
    if cfg.get('critic_observation') == 'actor_plus_reward_velocity_mean':
        return env.reward_velocity_window.prepare(env._command())
    # No privileged State; preserve a zero-width archive column for the pipeline.
    return np.empty(0,dtype=np.float32)
