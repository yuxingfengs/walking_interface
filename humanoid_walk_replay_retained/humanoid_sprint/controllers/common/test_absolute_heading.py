"""Absolute direction pipeline, wrapping, isolation and Actor/Critic tests."""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch
from test_unified_env import raw_state
from tasks.heading_correction import HeadingCorrectionTask, HeadingCorrectionConfig
from tasks.forward_walk_v1 import ForwardWalkTaskV1
from webots_platform.webots_runtime import absolute_heading
from contracts import Transition, CanonicalAction
from configs.runtime import load_config, algorithm_settings, project_path
from algorithms.ddpg import DDPGAlgorithm


class AbsoluteHeadingTests(unittest.TestCase):
    def test_world_rotation_not_imu(self):
        from com_estimator import rotation
        for angle in [-3.13, -.0837, 0., .0837, 3.13]:
            rot = rotation((0,0,1), angle) @ rotation((0,1,0), .2) @ rotation((1,0,0), .1)
            heading = absolute_heading(rot.reshape(-1))
            self.assertAlmostEqual(heading, angle)
            raw = replace(raw_state(), yaw=1.2, world_heading=heading)
            task = HeadingCorrectionTask(HeadingCorrectionConfig())
            obs = task.observe(raw, 0)
            self.assertAlmostEqual(obs.values[24], -angle)
            self.assertEqual(obs.values[:24], ForwardWalkTaskV1(task.config).observe(raw, 0).values)

    def test_wrap_and_missing_data(self):
        task = HeadingCorrectionTask(HeadingCorrectionConfig(target_heading_rad=math.pi-.01))
        obs = task.observe(replace(raw_state(), world_heading=-math.pi+.01), 0)
        self.assertAlmostEqual(obs.values[24], -.02)
        with self.assertRaises(ValueError): task.observe(raw_state(), 0)

    def test_reward_and_replay_use_same_error(self):
        task = HeadingCorrectionTask(HeadingCorrectionConfig(heading_reward_weight=.1))
        cfg = load_config()
        model = DDPGAlgorithm(**algorithm_settings(cfg, 'cpu', 'forward_walk_v1'))
        model.load_actor_initialization(project_path(cfg['initialization']['bc_checkpoint']), reset_optimizer=True)
        observations = [task.observe(replace(raw_state(), world_heading=a), 0) for a in [-.2, .2]]
        # Zero extra input weights preserve identical initial BC actions.
        torch.testing.assert_close(model.actor(torch.tensor(observations[0].values)), model.actor(torch.tensor(observations[1].values)))
        for obs in observations:
            reward = task.step_reward_from_transition(SimpleNamespace(distance_moved=.01, next_observation=obs, info={'fell':False}), None)
            self.assertAlmostEqual(reward, .08)
            action = CanonicalAction(tuple([0.]*10))
            model.observe(Transition(obs, action, reward, obs, False, False))
        states, actions, _, _, _ = model.replay_buffer.sample_batch(2)
        self.assertEqual(states.shape, (2,25))
        self.assertAlmostEqual(float(states[:,24].abs().mean()), .2, places=6)
        model.critic.eval()
        with torch.no_grad(): self.assertEqual(model.critic(states,actions).shape, (2,1))

if __name__ == '__main__': unittest.main()
