"""Hold the fixed forward target."""
from basewalk.omni_stages import WalkCommands


class ExperimentSchedule(WalkCommands):
    def command(self, error, speed):
        index = self.step // self.hold
        return self.target_at(index)


command_schedule = ExperimentSchedule
