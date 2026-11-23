"""Follow the forward speed sequence, then repeat from its configured index."""
from basewalk.omni_stages import WalkCommands


class ExperimentSchedule(WalkCommands):
    def command(self, error, speed):
        index = self.step // self.hold
        return self.target_at(index)


command_schedule = ExperimentSchedule
