"""Outer proportional heading hold; the policy follows the resulting yaw rate."""
import math
import numpy as np


def heading_command(reference, yaw, kp, max_rate):
    values = (reference, yaw, kp, max_rate)
    if not all(math.isfinite(value) for value in values) or kp <= 0 or max_rate <= 0:
        raise ValueError('Heading control requires finite angles and positive gains/limits')
    error = math.atan2(math.sin(reference-yaw), math.cos(reference-yaw))
    return max(-max_rate, min(max_rate, kp*error))


def turn_then_forward_phase(reference, yaw, walking, enter_tolerance_rad,
                            exit_tolerance_rad):
    """Update the turn/walk state with separate entry and exit thresholds."""
    values = (reference, yaw, enter_tolerance_rad, exit_tolerance_rad)
    if (not all(math.isfinite(value) for value in values) or
            enter_tolerance_rad <= 0 or exit_tolerance_rad < enter_tolerance_rad):
        raise ValueError('Turn/walk hysteresis requires finite ordered tolerances')
    error = abs(math.atan2(math.sin(reference-yaw), math.cos(reference-yaw)))
    return error <= (exit_tolerance_rad if walking else enter_tolerance_rad)


def turn_then_forward_command(reference, yaw, forward_speed, kp, max_rate,
                              forward_tolerance_rad, *, walking=None):
    """Turn in place, then walk forward while holding the requested heading."""
    error = math.atan2(math.sin(reference-yaw), math.cos(reference-yaw))
    wz = heading_command(reference, yaw, kp, max_rate)
    forward = abs(error) <= forward_tolerance_rad if walking is None else bool(walking)
    vx = forward_speed if forward else 0.
    return np.array([vx, 0., wz], dtype=np.float32)


def heading_blend_weights(reference, yaw, center_deg, width_deg):
    """Continuous turn/walk weights from wrapped absolute heading error."""
    values = (reference, yaw, center_deg, width_deg)
    if not all(math.isfinite(value) for value in values) or width_deg <= 0:
        raise ValueError('Heading blend requires finite values and positive width')
    error_deg = abs(math.degrees(math.atan2(math.sin(reference-yaw),
                                            math.cos(reference-yaw))))
    exponent = max(-60., min(60., (error_deg-center_deg)/width_deg))
    walk = 1. / (1. + math.exp(exponent))
    return 1.-walk, walk


def blended_turn_walk_command(reference, yaw, forward_speed, kp, max_rate,
                              center_deg, width_deg):
    """Scale forward speed continuously while heading feedback remains active."""
    turn_weight, walk_weight = heading_blend_weights(
        reference, yaw, center_deg, width_deg)
    wz = heading_command(reference, yaw, kp, max_rate)
    command = np.array([forward_speed*walk_weight, 0., wz], dtype=np.float32)
    return command, turn_weight, walk_weight
