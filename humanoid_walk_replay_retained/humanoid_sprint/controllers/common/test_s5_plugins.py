"""S5 tests for selectable Task and Algorithm plugins."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch


CONTROLLERS_DIR = Path(__file__).resolve().parent.parent
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from action_adapter import ActionAdapter
from algorithms import AlgorithmPlugin, load_algorithm
from algorithms.algorithm_plugin import AlgorithmUpdate
from contracts import (
    ACTION_DIM,
    ActorObservation,
    CanonicalAction,
    RawRobotState,
    STATE_DIM,
    Transition,
)
from envs.nao_locomotion_env import NaoLocomotionEnv
from tasks import load_task


def robot_state(*, time: float, x: float) -> RawRobotState:
    return RawRobotState(
        roll=0.0,
        yaw=0.0,
        gyro_x=0.0,
        gyro_y=0.0,
        center_of_mass_height=0.7,
        joint_positions=(0.0,) * 12,
        position=(x, 0.0, 0.7),
        simulation_time=time,
    )


class BackwardFakeRuntime:
    def __init__(self):
        self.states = (
            robot_state(time=0.00, x=0.00),
            robot_state(time=0.04, x=0.00),
            robot_state(time=0.08, x=0.00),
            robot_state(time=0.12, x=-0.04),
        )
        self.index = 0
        self.adapter = ActionAdapter()

    def hard_reset(self, settle_steps=64):
        self.index = 0
        return self.states[0]

    def step(self):
        self.index = min(self.index + 1, len(self.states) - 1)
        return 0

    def read_state(self):
        return self.states[self.index]

    def apply_action(self, action):
        return self.adapter.adapt(action)

    def get_position(self):
        return self.states[self.index].position


class S5PluginTests(unittest.TestCase):
    def algorithm(self, *, task_id="forward_walk_v1") -> AlgorithmPlugin:
        return load_algorithm(
            "algorithms.ddpg",
            "DDPGAlgorithm",
            {
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dim": 32,
                "actor_lr": 1e-5,
                "critic_lr": 1e-6,
                "device": "cpu",
                "gamma": 0.98,
                "tau": 1e-4,
                "batch_size": 2,
                "buffer_size": 8,
                "task_id": task_id,
            },
        )

    def test_ddpg_loads_through_algorithm_interface_and_updates(self):
        torch.manual_seed(1)
        algorithm = self.algorithm()
        observation = ActorObservation((0.0,) * STATE_DIM)
        action = algorithm.act(observation)
        self.assertEqual(tuple(action.shape), (ACTION_DIM,))

        for reward in (0.1, 0.2, 0.3):
            algorithm.observe(Transition(
                observation=observation,
                action=CanonicalAction((0.0,) * ACTION_DIM),
                reward=reward,
                next_observation=observation,
                terminated=False,
                truncated=False,
            ))
        update = algorithm.update()
        self.assertIsInstance(update, AlgorithmUpdate)
        self.assertEqual(algorithm.experience_size, 3)
        self.assertTrue(torch.isfinite(torch.tensor(update.actor_loss)))
        self.assertIn("bellman_gap", update.metrics)

    def test_checkpoint_task_contract_follows_selected_plugin(self):
        backward = self.algorithm(task_id="backward_walk_v1")
        forward = self.algorithm(task_id="forward_walk_v1")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "backward.pth"
            backward.save_checkpoint(checkpoint)
            metadata = backward.load_checkpoint(checkpoint)
            self.assertEqual(metadata["task_id"], "backward_walk_v1")
            self.assertEqual(metadata["algorithm_id"], "ddpg_v1")
            with self.assertRaises(ValueError):
                forward.load_checkpoint(checkpoint)

    def test_second_task_runs_through_same_unified_env(self):
        task = load_task(
            "tasks.backward_walk_v1",
            "BackwardWalkTaskV1",
            {
                "sequence_length": 35,
                "phase_mode": "sequence",
                "stop_x": -0.03,
            },
        )
        env = NaoLocomotionEnv(BackwardFakeRuntime(), task)
        env.reset()
        env.begin_rollout()
        step = env.step((0.0,) * ACTION_DIM, 0)
        self.assertEqual(task.task_id, "backward_walk_v1")
        self.assertAlmostEqual(step.distance_moved, 0.04)
        self.assertTrue(step.info["reached_goal"])
        self.assertFalse(step.terminated)


if __name__ == "__main__":
    unittest.main()
