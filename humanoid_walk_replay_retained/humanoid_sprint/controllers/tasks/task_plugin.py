"""Interface shared by selectable locomotion Task plugins."""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Mapping

from contracts import ACTION_ID, OBSERVATION_ID


class TaskPlugin(ABC):
    task_id = "task_plugin"
    observation_id = OBSERVATION_ID
    action_id = ACTION_ID
    config_type = None

    @classmethod
    def from_config(cls, config: Mapping):
        if cls.config_type is None:
            raise TypeError(f"{cls.__name__} does not declare config_type")
        return cls(cls.config_type(**dict(config)))

    def configure(self, **overrides) -> None:
        self.config = replace(self.config, **overrides)

    @abstractmethod
    def reset(self, raw_state):
        pass

    @abstractmethod
    def sync_position(self, raw_state) -> None:
        pass

    @abstractmethod
    def observe(self, raw_state, step_index: int):
        pass

    @abstractmethod
    def distance_moved(self, raw_state) -> float:
        pass

    @abstractmethod
    def termination(self, raw_state):
        pass

    @staticmethod
    @abstractmethod
    def reward(action, previous_action, distance_moved, fell, rollout_length, step_index):
        pass
