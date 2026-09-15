"""Current stronger heading Expert; the sine gait is its private base action."""
import math
from expert_policy import ExpertPolicy, ExpertMetadata

class StrongHeadingExpertPolicy(ExpertPolicy):
    metadata=ExpertMetadata(expert_id='strong_heading_stride_v1',uses_privileged_state=False,
        description='27D heading feedback, gain0.6, cap0.04, slew0.0005 per 40ms')

    def __init__(self, gain=.6, cap=.04):
        self.gain=gain
        self.cap=cap
        self.adjustment=0.

    def reset(self):
        self.adjustment=0.

    def act(self, observation, raw_state=None):
        if len(observation.values)!=27:
            raise ValueError('StrongHeadingExpertPolicy requires the 27D sin/cos/rate observation')
        error=math.atan2(observation.values[24],observation.values[25])
        target=max(-self.cap,min(self.cap,-self.gain*error))
        self.adjustment+=max(-.0005,min(.0005,target-self.adjustment))
        action=self._base_action(observation)
        angle=observation.values[0]*2*math.pi
        left=math.cos(angle)*self.adjustment
        right=-math.cos(angle+math.pi)*self.adjustment
        action[1]+=left; action[3]-=left; action[6]+=right; action[8]-=right
        return action

    @staticmethod
    def _base_action(observation):
        phase = observation.values[0]
        angle = phase * 2.0 * math.pi
        robot_height = 0.5
        shift_y = 0.3
        step_height = 0.4
        step_length = 0.2

        left_roll = math.sin(angle) * shift_y
        left_knee = (math.sin(angle) + 1.0) / 2.0 * step_height + robot_height
        left_hip_pitch = -left_knee / 2.0 + math.cos(angle) * step_length
        left_ankle_pitch = -left_knee / 2.0 - math.cos(angle) * step_length

        right_roll = left_roll
        right_knee = (math.sin(angle + math.pi) + 1.0) / 2.0 * step_height + robot_height
        right_hip_pitch = -right_knee / 2.0 + math.cos(angle + math.pi) * step_length
        right_ankle_pitch = -right_knee / 2.0 - math.cos(angle + math.pi) * step_length

        return [
            left_roll, left_hip_pitch, left_knee, left_ankle_pitch, -left_roll,
            right_roll, right_hip_pitch, right_knee, right_ankle_pitch, -right_roll,
        ]
