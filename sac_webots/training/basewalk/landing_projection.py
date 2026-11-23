"""Project touchdown joint targets and use the same feasible actions in DDPG."""
import math

import numpy as np
import torch

from .core import DEFAULT_POSE, MAXIMUM, MINIMUM


def rotation(axis, angle):
    x, y, z = axis
    k = angle.new_tensor(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    a = angle[..., None, None]
    return torch.eye(3, dtype=angle.dtype, device=angle.device) + a.sin()*k + (1-a.cos())*(k@k)


def apply(matrix, vector):
    return (matrix @ vector[..., None]).squeeze(-1)


def torso(gravity):
    gx, gy, gz = gravity.unbind(-1)
    return rotation((0, 1, 0), torch.atan2(gx, torch.hypot(gy, gz))) @ rotation((1, 0, 0), torch.atan2(-gy, -gz))


def axes(side):
    h = math.sqrt(.5)
    return ((0, h, h if side else -h), (1, 0, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0))


def leg_matrix(base, joints, side, last):
    out = base
    for axis, angle in zip(axes(side)[:last], joints.unbind(-1)):
        out = out @ rotation(axis, angle)
    return out


def sole_bottom(base, joints, side):
    sign = -1 if side else 1
    translations = ((0, sign*.05, -.085), (0,0,0), (0,0,0),
                    (0,0,-.1), (0,0,-.1029), (0,0,0))
    p = torch.zeros_like(joints[..., :3])
    m = base
    for translation, axis, angle in zip(translations, axes(side), joints.unbind(-1)):
        p = p + apply(m, joints.new_tensor(translation))
        m = m @ rotation(axis, angle)
    p = p + apply(m, joints.new_tensor((.02, sign*.007, -.041)))
    return p[..., 2] - (m[..., 2, :].abs()*joints.new_tensor((.0735,.0445,.005))).sum(-1)


def heading_error(lower, pitch):
    knee = torch.atan2(-lower[..., 0, 1], lower[..., 1, 1])
    toe = (lower @ rotation((0, 1, 0), pitch))[..., :, 0]
    error = torch.atan2(toe[..., 1], toe[..., 0]) - knee
    return torch.atan2(error.sin(), error.cos()).abs()


def wrap(angle):
    return torch.atan2(angle.sin(), angle.cos())


def landing_gate(obs, base, neutral, spec):
    measured = obs[..., 10:22] + neutral
    h = torch.stack([sole_bottom(base, measured[..., :6], 0),
                     sole_bottom(base, measured[..., 6:], 1)], -1)
    phase = torch.remainder(torch.atan2(obs[..., 9], obs[..., 8])/(2*math.pi), 1.)
    l0, l1 = spec['left_phase']
    r0, r1 = spec['right_phase']
    left = (phase >= l0) & (phase <= l1) if l0 <= l1 else (phase >= l0) | (phase <= l1)
    right = (phase >= r0) & (phase <= r1) if r0 <= r1 else (phase >= r0) | (phase <= r1)
    phases = torch.stack((left, right), -1)
    turn = (obs[..., 5].abs()<1e-8)&(obs[..., 7].abs()>1e-8)
    return phases & turn[..., None] & ((h-h.amin(dim=-1, keepdim=True))<=spec['near_other_sole_m'])


def joint_bounds(cfg):
    spec = cfg['landing_alignment']
    margin = cfg['environment']['joint_margin']
    return (np.maximum(MINIMUM, spec['motor_min']) + margin,
            np.minimum(MAXIMUM, spec['motor_max']) - margin)


def project_landing_actions(obs, action, cfg, *, diagnostics=False):
    spec = cfg['landing_alignment']
    tolerance = math.radians(spec['tolerance_deg'])
    scale = cfg['environment']['action_scale']
    neutral = action.new_tensor(DEFAULT_POSE)
    lower, upper = joint_bounds(cfg)
    lo = torch.maximum(action.new_tensor(lower), neutral-scale)
    hi = torch.minimum(action.new_tensor(upper), neutral+scale)
    raw = action.clamp(-1., 1.)
    hip = ((raw[..., 0]+raw[..., 6])*.5)*scale + neutral[0]
    hip = hip.clamp(torch.maximum(lo[0],lo[6]), torch.minimum(hi[0],hi[6]))
    q = (neutral + scale*raw).clamp(lo,hi)
    q = torch.cat((hip[...,None],q[...,1:6],hip[...,None],q[...,7:]),-1)
    base = torso(obs[..., :3])
    gate = landing_gate(obs,base,neutral,spec)
    result, errors = [], []
    for side in range(2):
        i = side*6
        leg = q[..., i:i+6]
        lower = leg_matrix(base, leg, side, 4)
        pitch = leg[..., 4]
        error = heading_error(lower,pitch)
        needed = gate[..., side] & (error>tolerance*.999)
        # Level the forward axis, then find the earliest point reaching the tolerance.
        level = torch.atan2(lower[..., 2, 0],lower[..., 2, 2])
        level = torch.where((level-pitch).cos()<0,wrap(level+math.pi),level).clamp(lo[i+4],hi[i+4])
        can_use_pitch = heading_error(lower,level)<tolerance*.998
        # Branch comparisons select a scalar fraction; backprop still flows
        # through the original and feasible endpoint joint targets.
        with torch.no_grad():
            a = torch.zeros_like(pitch)
            b = torch.ones_like(pitch)
            for _ in range(14):
                mid = (a+b)*.5
                outside = heading_error(lower,pitch+mid*(level-pitch))>tolerance*.998
                a = torch.where(outside,mid,a)
                b = torch.where(outside,b,mid)
        corrected_pitch = torch.where(needed&can_use_pitch,pitch+b*(level-pitch),pitch)
        cap = spec.get('max_ankle_pitch_correction_deg')
        if cap is not None:
            corrected_pitch = torch.where(needed & ~can_use_pitch, level, corrected_pitch)
            limit = math.radians(cap)
            change = (corrected_pitch - pitch).clamp(-limit, limit)
            bounded_pitch = pitch + change
            improves = heading_error(lower, bounded_pitch) <= error
            corrected_pitch = torch.where(improves, bounded_pitch, pitch)
        # Legacy configurations can also level the knee axis when pitch is blocked.
        hip_yaw = base @ rotation(axes(side)[0],leg[..., 0])
        roll = leg[..., 1]
        level_knee = torch.atan2(-hip_yaw[..., 2, 1],hip_yaw[..., 2, 2])
        level_knee = torch.where((level_knee-roll).cos()<0,wrap(level_knee+math.pi),level_knee)
        level_knee = level_knee.clamp(lo[i+1],hi[i+1])
        corrected_roll = roll if cap is not None else torch.where(needed&~can_use_pitch,level_knee,roll)
        fixed = torch.stack((leg[..., 0],corrected_roll,leg[..., 2],leg[..., 3],
                             corrected_pitch,leg[..., 5]),-1)
        result.append(fixed)
        with torch.no_grad():
            errors.append(heading_error(leg_matrix(base, fixed, side, 4), corrected_pitch))
    projected = (torch.cat(result, -1) - neutral) / scale
    error = torch.stack(errors, -1)
    invalid = ~torch.isfinite(error)
    if spec.get('max_ankle_pitch_correction_deg') is None:
        invalid = invalid | (error > tolerance)
    if torch.any(gate & invalid):
        raise ValueError('No feasible landing Action within the knee-foot yaw tolerance')
    if diagnostics:
        return projected, gate, error
    return projected
