"""NAO sole headings from torso attitude and measured leg joints."""
import numpy as np

from .core import DEFAULT_POSE


def rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.asarray(angle)[..., None, None]
    return np.eye(3) + np.sin(angle)*skew + (1 - np.cos(angle))*(skew @ skew)


def relative_headings(gravity, joints):
    # Resolve torso pitch/roll with heading zero. The world heading increment
    # is added separately, so a reset origin is not required.
    gx, gy, gz = np.moveaxis(np.asarray(gravity), -1, 0)
    pitch = np.arctan2(gx, np.hypot(gy, gz))
    roll = np.arctan2(-gy, -gz)
    r = rotation((0, 1, 0), pitch) @ rotation((1, 0, 0), roll)
    joints = np.asarray(joints)
    headings = []
    h = np.sqrt(0.5)
    # Axes match the NAO PROTO; fixed sole-sensor yaw offsets cancel in differences.
    for i, sign in enumerate((-1, 1)):
        leg = r
        axes = ((0, h, sign*h), (1, 0, 0), (0, 1, 0), (0, 1, 0),
                (0, 1, 0), (1, 0, 0))
        for k, axis in enumerate(axes):
            leg = leg @ rotation(axis, joints[..., i*6+k])
        headings.append(np.arctan2(leg[..., 1, 0], leg[..., 0, 0]))
    return np.stack(headings, axis=-1)


def reverse_cost(before_headings, after_headings, heading_delta, before_contact, after_contact, dt, command):
    delta = np.asarray(after_headings) - before_headings + np.asarray(heading_delta)[..., None]
    direction = np.sign(np.asarray(command))[..., None]
    yaw_step = np.arctan2(np.sin(delta), np.cos(delta))
    reverse_rate = np.maximum(0, -direction*yaw_step/np.asarray(dt)[..., None])
    return np.sum(np.square(reverse_rate)*(np.asarray(before_contact) & after_contact), axis=-1)


def observation_cost(obs, nxt, heading_delta, before_contact, after_contact, dt):
    return reverse_cost(relative_headings(obs[:, :3], obs[:, 10:22]+DEFAULT_POSE),
                        relative_headings(nxt[:, :3], nxt[:, 10:22]+DEFAULT_POSE),
                        heading_delta, before_contact, after_contact, dt, obs[:, 7])
