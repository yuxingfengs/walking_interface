"""Integrate actual/commanded world displacement and unwrapped yaw increments."""
import math
import numpy as np


def segment_rewards(actual, target, durations, config):
    durations = np.asarray(durations, dtype=float)
    if len(durations)==0 or not np.isfinite(durations).all() or np.any(durations<=0):
        raise ValueError('Invalid cycle durations')
    error = (np.asarray(actual,dtype=float)-np.asarray(target,dtype=float)).sum(axis=0)/durations.sum()
    if error.shape!=(3,) or not np.isfinite(error).all():
        raise ValueError('Invalid cycle integrals')
    xy=config['weight_xy']*math.exp(-float(error[:2]@error[:2])/config['sigma_xy'])
    if config.get('yaw_mode') == 'linear_absolute':
        # Absolute increment error before integration: opposite errors cannot cancel.
        delta = np.asarray(actual,dtype=float)-np.asarray(target,dtype=float)
        yaw = -config['weight_yaw']*float(np.abs(delta[:,2]).sum())/durations.sum()
    else:
        yaw=config['weight_yaw']*math.exp(-float(error[2]**2)/config['sigma_yaw'])
    return durations*xy, durations*yaw


def trajectory_rewards(actual, target, durations, modes, boundaries, config):
    if config.get('temporal_mode') == 'per_step':
        dt=np.asarray(durations,dtype=float)
        if not np.isfinite(dt).all() or np.any(dt<=0):raise ValueError('Invalid durations')
        if len(dt)==0:return np.zeros(0)
        error=(np.asarray(actual,dtype=float)-np.asarray(target,dtype=float))/dt[:,None]
        result = dt*(config['weight_xy']*np.exp(-np.square(error[:,:2]).sum(axis=1)/config['sigma_xy'])
                   -config['weight_yaw']*np.abs(error[:,2]))
        if 'walk_forward_only_weight' in config:
            walk=np.asarray([m==2 or m=='walk' for m in modes])
            target_xy=np.asarray(target,dtype=float)[:,:2]
            lengths=np.linalg.norm(target_xy,axis=1)
            if np.any(lengths[walk]<=0):raise ValueError('Walk requires a nonzero forward command to resolve source heading')
            # In this task Walk has vy_cmd=0 and vx_cmd>0. Its world command
            # direction is the source body-forward axis, not the goal ray.
            forward=np.sum(np.asarray(actual,dtype=float)[walk,:2]*target_xy[walk],axis=1)/lengths[walk]
            result[walk]=config['walk_forward_only_weight']*forward
        return result
    result=np.zeros(len(durations))
    start=0
    for i in range(len(result)):
        end=bool(boundaries[i]) or i==len(result)-1 or modes[i+1]!=modes[i]
        if end:
            xy,yaw=segment_rewards(actual[start:i+1],target[start:i+1],durations[start:i+1],config)
            result[start:i+1]=xy+yaw
            start=i+1
    return result
