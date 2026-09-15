import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from algorithms.ddpg import DDPGAlgorithm
from contracts import ActorObservation, CanonicalAction, Transition
from world_reload_state import save_continuation, load_continuation, restore_rng
from configs.runtime import load_config, runtime_settings


class WorldReloadTests(unittest.TestCase):
    def test_full_learner_replay_optimizer_and_rng_survive_reload(self):
        algorithm = DDPGAlgorithm(state_dim=18, action_dim=10, hidden_dim=256,
                                  actor_lr=1e-5, critic_lr=1e-6, buffer_size=32, batch_size=2,
                                  device='cpu', gamma=.98, tau=1e-4)
        observation = ActorObservation(tuple([.1]*18))
        for _ in range(4):
            algorithm.observe(Transition(observation, CanonicalAction(tuple([.1]*10)),
                                         .1, observation, False, False))
        algorithm.update()
        metadata = dict(continuous_training=dict(env_steps=350, steps_since_update=0), global_step=1)
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)/'effective_config.json'
            snapshot.write_text(json.dumps(dict(run_id='test')))
            save_continuation(snapshot, algorithm, metadata)
            expected = (random.random(), np.random.random(), torch.rand(3))
            bundle = load_continuation(snapshot)
            restored = bundle['algorithm']
            restore_rng(bundle['rng'])
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.random(), expected[1])
            self.assertTrue(torch.equal(torch.rand(3), expected[2]))
            self.assertEqual(restored.replay_buffer.buffer, algorithm.replay_buffer.buffer)
            for name in ('actor', 'critic', 'actor_target', 'critic_target'):
                for a,b in zip(getattr(algorithm,name).parameters(),getattr(restored,name).parameters()):
                    self.assertTrue(torch.equal(a,b))
            self.assertEqual(restored.actor_optimizer.param_groups[0]['lr'], 1e-5)
            self.assertEqual(restored.critic_optimizer.param_groups[0]['lr'], 1e-6)
            first = next(restored.actor.parameters())
            self.assertIs(first, restored.actor_optimizer.param_groups[0]['params'][0])
            original_first = next(algorithm.actor.parameters())
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(restored.actor_optimizer.state[first][key],
                                            algorithm.actor_optimizer.state[original_first][key]))
            self.assertEqual(restored.actor_update_count, algorithm.actor_update_count)
            self.assertEqual(bundle['metadata'], metadata)
            self.assertIsNone(load_continuation(snapshot))

    def test_evaluation_keeps_original_reset_protocol(self):
        config = load_config()
        config['initialization']['training_reset_mode'] = 'world_reload'
        self.assertEqual(runtime_settings(config, mode='train')['reset_mode'], 'world_reload')
        self.assertEqual(runtime_settings(config, mode='eval')['reset_mode'], 'legacy')
