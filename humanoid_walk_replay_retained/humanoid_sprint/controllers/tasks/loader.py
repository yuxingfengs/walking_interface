"""Configuration-driven Task plugin loading."""

import importlib
from typing import Mapping

from tasks.task_plugin import TaskPlugin


def load_task(module_name: str, class_name: str, config: Mapping) -> TaskPlugin:
    module = importlib.import_module(module_name)
    task_class = getattr(module, class_name)
    if not issubclass(task_class, TaskPlugin):
        raise TypeError(f"{module_name}.{class_name} must implement TaskPlugin")
    return task_class.from_config(config)
