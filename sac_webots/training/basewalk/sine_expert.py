"""Physical phase-only sine experts, ported from the validated v6 expert bank.

These controllers do not use state feedback to recover from disturbances.
"""
import math
import numpy as np


def sine_targets(phase):
    angle = phase * 2 * math.pi
    roll = math.sin(angle) * .3
    result = []
    for a in (angle, angle + math.pi):
        knee = (math.sin(a) + 1) / 2 * .4 + .5
        result.extend([0., roll, -knee / 2 + math.cos(a) * .2,
                       knee, -knee / 2 - math.cos(a) * .2, -roll])
    return np.asarray(result, dtype=np.float32)


def directional_targets(phase, parameters):
    forward, lateral, differential, yaw_amplitude, sway, bias = parameters
    yaw = yaw_amplitude * math.cos(2 * math.pi * phase)
    q = sine_targets(phase).copy()
    sway_delta = (sway - .3) * math.sin(2 * math.pi * phase)
    for side, offset in enumerate((0, 6)):
        c = math.cos(2 * math.pi * phase + side * math.pi)
        pitch = (forward - .2 + (1 if side == 0 else -1) * differential) * c
        roll = -lateral * c + sway_delta + bias
        q[offset + 2] += pitch
        q[offset + 4] -= pitch
        q[offset + 1] += roll
        q[offset + 5] -= roll
        q[offset] = yaw
        q[offset + 2] -= yaw / math.sqrt(2)
    return q


def expert_targets(phase, elapsed, parameters):
    ramp = min(1., elapsed / 3.2)
    ramp = ramp * ramp * (3 - 2 * ramp)
    base = sine_targets(phase)
    return base + ramp * (directional_targets(phase, parameters) - base)


def turn_then_forward_targets(phase, elapsed, parameters, command, max_yaw_rate=.3):
    """Physical expert: differential stepping while turning, normal gait once aligned."""
    turn_differential, hip_yaw_amplitude = parameters
    wz = float(command[2])
    if abs(wz) > .02:
        strength = max(-1., min(1., wz/max_yaw_rate))
        gait = (0., 0., turn_differential*strength,
                hip_yaw_amplitude*strength, .3, 0.)
    else:
        gait = (.2, 0., 0., 0., .3, 0.)
    return expert_targets(phase, elapsed, gait)


def three_stage_targets(phase, elapsed, parameters, stage, command, max_yaw_rate=.3):
    """Turn in place, step in place, then walk forward under explicit commands."""
    turn_differential, hip_yaw_amplitude, walk_forward, hold_knee_lift = parameters
    if stage == 'turn':
        strength = max(-1., min(1., float(command[2])/max_yaw_rate))
        gait = (0., 0., turn_differential*strength,
                hip_yaw_amplitude*strength, .3, 0.)
    elif stage == 'hold':
        # forward=0 cancels the sagittal swing in sine_targets while retaining
        # alternating knee lift. Add explicit flexion so this is a visible,
        # identifiable skill rather than a brief neutral-gait transition.
        gait = (0., 0., 0., 0., .3, 0.)
    elif stage == 'walk':
        # Keep walking while applying the live heading-controller command.
        # Previously wz was present in the Observation but ignored by the
        # physical walk target, so small gait asymmetries accumulated unchecked.
        strength = max(-1., min(1., float(command[2])/max_yaw_rate))
        gait = (walk_forward, 0., turn_differential*strength,
                hip_yaw_amplitude*strength, .3, 0.)
    else:
        raise ValueError('Unknown three-stage expert phase: '+str(stage))
    target = expert_targets(phase, elapsed, gait)
    if stage == 'hold':
        ramp = min(1., elapsed/3.2)
        ramp = ramp*ramp*(3-2*ramp)
        angle = 2*math.pi*phase
        for side, offset in enumerate((0, 6)):
            lift = hold_knee_lift*max(0., math.sin(angle+side*math.pi))*ramp
            target[offset+2] -= lift/2
            target[offset+3] += lift
            target[offset+4] -= lift/2
    return target
