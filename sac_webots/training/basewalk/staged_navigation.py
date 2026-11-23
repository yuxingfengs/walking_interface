"""Staged command and reward with 1-degree entry / 3-degree exit hysteresis."""
import math
import numpy as np
from .heading_control import heading_command
from .stationary_reward import stationary_terms


def command(reference, yaw, speed, cfg, stand=False, walking=False):
    if stand:
        return np.zeros(3, dtype=np.float32), 0., 0.
    from .heading_control import turn_then_forward_phase
    e=cfg['environment']
    walk = bool(walking and e.get('latch_walk_until_reset', False)) or turn_then_forward_phase(reference,yaw,walking,
        math.radians(e['turn_to_walk_tolerance_deg']),math.radians(e['walk_to_turn_tolerance_deg']))
    if e.get('heading_command_mode') == 'fixed_turn_zero_walk':
        error = math.atan2(math.sin(reference-yaw), math.cos(reference-yaw))
        wz = 0. if walk else math.copysign(e['heading_max_rate'], error)
    else:
        wz = heading_command(reference,yaw,e['heading_kp'],e['heading_max_rate'])
    return np.array([speed if walk else 0.,0.,wz],
                    dtype=np.float32),float(not walk),float(walk)


def reward(old_position, position, reference, old_yaw, yaw, source_command,
           velocity, net_speed, ready, cfg, dt, fell=False, stand=False):
    new_heading = yaw
    target = np.asarray(source_command, dtype=float)
    actual = np.asarray(velocity, dtype=float)
    p = cfg['velocity_objective']
    if target[0]>1e-8 and 'walk_forward_only_weight' in p:
        delta=np.asarray(position)[:2]-np.asarray(old_position)[:2]
        forward=float(np.array([math.cos(old_yaw),math.sin(old_yaw)])@delta)
        total=p['walk_forward_only_weight']*forward
        return total, dict(tracking_xy=0.,tracking_yaw=0.,walk_forward_progress=total,
            turn_weight=0.,walk_weight=1.,stand_weight=0.,stand_total=0.,
            turn_total=0.,walk_total=total,turn_weighted=0.,walk_weighted=total,turn_hold=0.)
    xy = p['weight_xy']*math.exp(-float(np.square(actual[:2]-target[:2]).sum())/p['sigma_xy'])*dt
    if p.get('yaw_mode') == 'linear_absolute':
        yaw = -p['weight_yaw']*abs(float(actual[2]-target[2]))*dt
    else:
        yaw = p['weight_yaw']*math.exp(-float((actual[2]-target[2])**2)/p['sigma_yaw'])*dt
    if p.get('temporal_mode') == 'per_step':
        # Same source-frame command/increment definition as stored expert data.
        delta=np.asarray(position)[:2]-np.asarray(old_position)[:2]
        c,s=math.cos(old_yaw),math.sin(old_yaw)
        actual_xy=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])/dt
        actual_w=math.atan2(math.sin(new_heading-old_yaw),math.cos(new_heading-old_yaw))/dt
        xy=p['weight_xy']*math.exp(-float(np.square(actual_xy-target[:2]).sum())/p['sigma_xy'])*dt
        yaw=-p['weight_yaw']*abs(actual_w-target[2])*dt
    walk = float(target[0]>1e-8)
    total = xy+yaw
    return total, dict(tracking_xy=xy,tracking_yaw=yaw,turn_weight=1.-walk,walk_weight=walk,
                       stand_weight=0.,stand_total=0.,turn_total=total,walk_total=total,
                       turn_weighted=(1.-walk)*total,walk_weighted=walk*total,turn_hold=0.)
