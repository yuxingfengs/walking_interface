"""Alternate ordinary coverage with pending failed-angle retries."""
from collections import deque
import numpy as np


class FailureRetry:
    def __init__(self, direction, seed):
        self.angles = list(direction['angles_deg'])
        self.speed = direction['speed_m_s']
        self.rng = np.random.default_rng(seed)
        self.pending = []
        self.failed = deque()
        self.last_angle = None
        self.retry_next = False
        self.stage_steps = 0
        self.completed = False
        self.budget_exhausted = False

    def goal_config(self, direction, angle):
        return dict(direction)

    def next(self, step=None):
        if self.retry_next and self.failed:
            angle = self.failed[0]
            self.failed.rotate(-1)
            self.retry_next = False
        else:
            reserved = self.failed[0] if self.failed else None
            choices = [a for a in self.angles if a != reserved and a != self.last_angle]
            if not choices:
                # With two angles, the spacer may repeat the last ordinary task,
                # but it must still differ from the pending retry target.
                choices = [a for a in self.angles if a != reserved]
            if not choices:
                raise ValueError('Alternating failed-angle retry requires at least two angles')
            angle = self.rng.choice(choices).item()
            self.retry_next = bool(self.failed)
        self.last_angle = angle
        return angle, [self.speed,0.,0.]

    def record_outcome(self, angle, success):
        if success:
            if angle in self.failed:
                self.failed.remove(angle)
        elif angle not in self.failed:
            self.failed.append(angle)
            # First failure must be followed by a different ordinary attempt.
            if len(self.failed) == 1:
                self.retry_next = False

    def record_strict_success(self, angle):
        return False

    def reset_focus_history(self):
        pass  # Recovery must not erase pending failed-angle retries.

    def sampling_state(self, angle):
        return dict(failed_angles=list(self.failed), retry_next=self.retry_next,
                    task_sampling='random_other_then_failed_angle')
