"""Unified environment joining WebotsRuntime, Task, and shared contracts."""

from dataclasses import dataclass
from typing import Dict

from contracts import ActorObservation, CanonicalAction, Transition


@dataclass(frozen=True)
class EnvStep:
    observation: ActorObservation
    action: CanonicalAction
    next_observation: ActorObservation
    distance_moved: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class NaoLocomotionEnv:
    def __init__(self, runtime, task):
        self.runtime = runtime
        self.task = task
        self.current_observation = None
        self.last_raw_state = None

    def initialize(self) -> None:
        self.runtime.initialize()

    def reset(self, settle_steps: int = 64) -> ActorObservation:
        phase_origin_state = self.runtime.hard_reset(settle_steps=settle_steps)
        if hasattr(self.runtime, "reset_progress_estimator"):
            self.runtime.reset_progress_estimator()
        self.task.reset(phase_origin_state)
        self.runtime.step()
        self.last_raw_state = self.runtime.read_state()
        self.current_observation = self.task.observe(self.last_raw_state, 0)
        return self.current_observation

    def begin_rollout(self, step_index: int = 0) -> ActorObservation:
        self.runtime.step()
        self.last_raw_state = self.runtime.read_state()
        self.task.sync_position(self.last_raw_state)
        self.current_observation = self.task.observe(self.last_raw_state, step_index)
        return self.current_observation

    def step(self, action, step_index: int) -> EnvStep:
        if self.current_observation is None:
            raise RuntimeError("environment must be reset before step")
        values = action.detach().cpu().tolist() if hasattr(action, "detach") else action
        canonical_action = self.runtime.apply_action(values)
        self.runtime.step()
        raw_state = self.runtime.read_state()
        next_observation = self.task.observe(raw_state, step_index + 1)
        distance = self.task.distance_moved(raw_state)
        fell, reached_goal = self.task.termination(raw_state)
        result = EnvStep(
            observation=self.current_observation,
            action=canonical_action,
            next_observation=next_observation,
            distance_moved=distance,
            # Preserve the S0 Replay Buffer contract: only a fall is terminal.
            # Reaching stop_x ends the rollout and triggers a reset in the
            # controller, but its last transition remains bootstrappable.
            terminated=bool(fell),
            truncated=False,
            info={
                "estimated_forward": raw_state.estimated_forward,
                "odometry_valid": raw_state.odometry_valid,
                "fell": bool(fell),
                "reached_goal": bool(reached_goal),
                "position": raw_state.position,
                "center_of_mass_height": raw_state.center_of_mass_height,
            },
        )
        self.current_observation = next_observation
        self.last_raw_state = raw_state
        return result

    def finalize_transition(
        self,
        step: EnvStep,
        previous_action,
        *,
        rollout_fell: bool,
        rollout_length: int,
        step_index: int,
    ) -> Transition:
        previous_values = (
            previous_action.detach().cpu().tolist()
            if hasattr(previous_action, "detach") else previous_action
        )
        reward = self.task.reward(
            step.action.values,
            previous_values,
            step.distance_moved,
            rollout_fell,
            rollout_length,
            step_index,
        )
        return Transition(
            observation=step.observation,
            action=step.action,
            reward=reward,
            next_observation=step.next_observation,
            terminated=step.terminated,
            truncated=step.truncated,
        )

    def step_transition(self, step: EnvStep, previous_action) -> Transition:
        """Store the pre-reset successor; administrative resets bootstrap from it."""
        transition_reward = getattr(self.task, 'step_reward_from_transition', None)
        if transition_reward is not None:
            reward = transition_reward(step, previous_action)
        else:
            reward = self.task.step_reward(step.action.values, previous_action,
                                           step.distance_moved, step.info['fell'])
        return Transition(observation=step.observation, action=step.action,
                          reward=reward, next_observation=step.next_observation,
                          terminated=bool(step.info['fell']),
                          truncated=False)

    @property
    def position(self):
        return self.runtime.get_position()

    @property
    def center_of_mass_height(self) -> float:
        if self.last_raw_state is None:
            return self.runtime.read_state().center_of_mass_height
        return self.last_raw_state.center_of_mass_height
