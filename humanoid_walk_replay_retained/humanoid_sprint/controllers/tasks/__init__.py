"""Task plugins."""

from tasks.loader import load_task
from tasks.task_plugin import TaskPlugin

__all__ = ("TaskPlugin", "load_task")

