"""Causal heading tracking over a bounded time window."""
from collections import deque
import math


class TurnHeadingWindow:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Turn heading window must be positive and finite')
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.steps = deque()
        self.elapsed = 0.0
        self.error = 0.0
        self.command = None

    def advance(self, command, yaw_delta, dt):
        if command == 0:
            self.reset()
            return 0.0
        if not all(map(math.isfinite, (command, yaw_delta, dt))) or dt <= 0:
            raise ValueError('Heading tracking requires finite samples and positive dt')
        if self.command != command:
            self.reset()
            self.command = command
        delta = command*dt - yaw_delta
        self.steps.append((dt, delta))
        self.elapsed += dt
        self.error += delta
        while self.elapsed - self.steps[0][0] >= self.seconds:
            duration, correction = self.steps.popleft()
            self.elapsed -= duration
            self.error -= correction
        duration, correction = self.steps[0]
        return self.error - correction*max(0.0, self.elapsed-self.seconds)/duration


def sequence_errors(seconds, commands, yaw_deltas, durations, episode_starts):
    window = TurnHeadingWindow(seconds)
    for command, yaw_delta, dt, start in zip(commands, yaw_deltas, durations, episode_starts):
        if start:
            window.reset()
        yield window.advance(float(command), float(yaw_delta), float(dt))
