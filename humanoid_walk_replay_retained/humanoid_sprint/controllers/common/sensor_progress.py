"""Contact kinematic odometry and monotonic progress watchdog (no world pose)."""
from collections import deque
import numpy as np
from com_estimator import rotation


class ContactOdometry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.heading = None
        self.previous = None
        self.previous_contacts = np.zeros(2, dtype=bool)
        self.forward = 0.

    def observe(self, soles, roll, pitch, yaw, contacts):
        soles = np.asarray(soles, dtype=float)
        if soles.shape != (2, 3) or not np.isfinite(soles).all() or not np.isfinite([roll,pitch,yaw,*contacts]).all():
            self.previous = None
            self.previous_contacts[:] = False
            return self.forward, False
        if self.heading is None:
            self.heading = yaw
        orientation = rotation((0,0,1), yaw-self.heading) @ rotation((0,1,0),pitch) @ rotation((1,0,0),roll)
        feet = soles @ orientation.T
        touching = np.asarray(contacts, dtype=float) > 0
        continuing = touching & self.previous_contacts
        valid = self.previous is not None and bool(continuing.any())
        if valid:
            # A stationary support foot moves backwards relative to a forward torso.
            delta = -(feet[continuing]-self.previous[continuing]).mean(axis=0)
            # Reject gross contact discontinuities; never substitute truth.
            valid = bool(np.linalg.norm(delta) <= .05)
            if valid:
                self.forward += float(delta[0])
        self.previous = feet
        self.previous_contacts = touching
        return self.forward, valid


class ForwardProgressWatchdog:
    def __init__(self, window_steps=35, minimum_progress=.02):
        if window_steps < 1 or minimum_progress <= 0:
            raise ValueError('Progress window and minimum progress must be positive')
        self.samples = deque(maxlen=window_steps)
        self.minimum_progress = minimum_progress
        self.reset()

    def reset(self):
        self.samples.clear()
        self.anchor = 0.
        self.filtered_forward = 0.
        self.wait_steps = 0

    def observe(self, forward, valid=True):
        self.wait_steps += 1
        if valid and np.isfinite(forward):
            self.samples.append(float(forward))
            self.filtered_forward = float(np.mean(self.samples))
            if self.filtered_forward >= self.anchor+self.minimum_progress:
                self.anchor = self.filtered_forward
                self.wait_steps = 0
        return self.wait_steps
