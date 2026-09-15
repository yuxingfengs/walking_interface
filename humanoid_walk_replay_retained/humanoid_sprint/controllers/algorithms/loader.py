"""Configuration-driven Algorithm plugin loading."""

import importlib
from typing import Mapping

from algorithms.algorithm_plugin import AlgorithmPlugin


def load_algorithm(module_name: str, class_name: str, config: Mapping):
    module = importlib.import_module(module_name)
    algorithm_class = getattr(module, class_name)
    algorithm = algorithm_class(**dict(config))
    if not isinstance(algorithm, AlgorithmPlugin):
        raise TypeError(
            f"{module_name}.{class_name} must implement AlgorithmPlugin"
        )
    return algorithm
