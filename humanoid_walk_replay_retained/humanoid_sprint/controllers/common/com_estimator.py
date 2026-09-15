"""Sensor-only, quasi-static NAO V5 COM relative to the supporting sole(s).

URDF supplies fixed geometry/masses, never live simulator transforms. Coordinates
are gravity aligned with torso heading: x forward, y left, z up, in metres.
The wrist masses include the hands; tiny URDF gripper dummy masses are omitted.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        r, p, y = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
        transform[:3, :3] = rotation((0, 0, 1), y) @ rotation((0, 1, 0), p) @ rotation((1, 0, 0), r)
    return transform


class NaoCOMEstimator:
    def __init__(self):
        root = ET.parse(Path(__file__).with_name('models') / 'nao_v50.urdf').getroot()
        self.masses = {}
        for link in root.findall('link'):
            inertial = link.find('inertial')
            if inertial is not None:
                mass = float(inertial.find('mass').get('value'))
                if mass > 1e-5:
                    self.masses[link.get('name')] = (mass, origin(inertial.find('origin'))[:, 3])
        joints = {j.find('child').get('link'): j for j in root.findall('joint')}
        needed = set(self.masses) | {'l_sole', 'r_sole'}
        for link in list(needed):
            while link in joints:
                link = joints[link].find('parent').get('link')
                needed.add(link)
        self.joints = []
        self.joint_names = []
        for child, j in joints.items():
            if child not in needed:
                continue
            moving = j.get('type') != 'fixed'
            name = j.get('name')
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if moving else None
            self.joints.append((name, j.find('parent').get('link'), child, origin(j.find('origin')), axis))
            if moving:
                self.joint_names.append(name)
        self.total_mass = sum(mass for mass, _ in self.masses.values())

    def kinematics(self, angles):
        if not all(name in angles and np.isfinite(angles[name]) for name in self.joint_names):
            raise ValueError('COM requires finite measured angles for all model joints')
        transforms = {'base_link': np.eye(4)}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, parent, child, fixed, axis in pending:
                if parent not in transforms:
                    remaining.append((name, parent, child, fixed, axis))
                    continue
                motion = np.eye(4)
                if axis is not None:
                    # Webots exposes both hip encoders; use each measurement,
                    # rather than URDF's physical-robot mimic assumption.
                    motion[:3, :3] = rotation(axis, angles[name])
                transforms[child] = transforms[parent] @ fixed @ motion
            if len(remaining) == len(pending):
                raise ValueError('Disconnected COM model')
            pending = remaining
        com = sum(mass * (transforms[name] @ local)[:3]
                  for name, (mass, local) in self.masses.items()) / self.total_mass
        return com, np.array([transforms[name][:3, 3] for name in ('l_sole', 'r_sole')])

    def estimate(self, angles, roll, pitch, contacts):
        if not np.isfinite([roll, pitch, *contacts]).all():
            return (0., 0., 0.), 0.
        weights = np.asarray(contacts, dtype=float) > 0
        if not weights.any():
            return (0., 0., 0.), 0.
        com, soles = self.kinematics(angles)
        support = soles[weights].mean(axis=0)
        upright = rotation((0, 1, 0), pitch) @ rotation((1, 0, 0), roll)
        return tuple(float(v) for v in upright @ (com - support)), 1.
