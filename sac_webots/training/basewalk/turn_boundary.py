"""Track displacement within each consecutive source-command Turn segment."""
import math


class TurnBoundary:
    def __init__(self, limit):
        self.limit = float(limit)
        if not math.isfinite(self.limit) or self.limit < 0:
            raise ValueError('Turn reset distance must be finite and nonnegative')
        self.reset()

    def reset(self):
        self.origin = None
        self.distance = 0.

    def begin(self, turning, position):
        if not turning:
            self.reset()
        elif self.origin is None:
            self.origin = (float(position[0]), float(position[1]))

    def exceeded(self, position):
        if self.origin is None or self.limit == 0:
            return False
        self.distance = math.hypot(float(position[0])-self.origin[0],
                                   float(position[1])-self.origin[1])
        return self.distance > self.limit
