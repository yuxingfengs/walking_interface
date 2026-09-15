"""Exercise real optimization across the warmup and resume boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from algorithms.ddpg import DDPGAlgorithm
from contracts import ActorObservation, CanonicalAction, Transition


class CriticWarmupTests(unittest.TestCase):
    def algorithm(self, warmup=2):
        torch.manual_seed(17)
        model = DDPGAlgorithm(state_dim=18, action_dim=10, hidden_dim=32,
            actor_lr=1e-5, critic_lr=1e-6, device='cpu', gamma=.98, tau=1e-4,
            batch_size=4, buffer_size=16, critic_warmup_updates=warmup)
        for i in range(8):
            obs = ActorObservation(tuple(torch.randn(18).tolist()))
            nxt = ActorObservation(tuple(torch.randn(18).tolist()))
            action = CanonicalAction(tuple(model.act(obs).tolist()))
            model.observe(Transition(obs, action, .1 + i * .03, nxt, False, False))
        return model

    def test_actor_and_target_are_exactly_frozen_then_resume(self):
        model = self.algorithm()
        actor = copy.deepcopy(model.actor.state_dict())
        target = copy.deepcopy(model.actor_target.state_dict())
        critic = copy.deepcopy(model.critic.state_dict())
        for index in range(2):
            result = model.update()
            self.assertFalse(result.metrics['actor_updated'])
            self.assertEqual(model.critic_warmup_completed, index + 1)
            self.assertTrue(all(torch.equal(v, model.actor.state_dict()[k]) for k, v in actor.items()))
            self.assertTrue(all(torch.equal(v, model.actor_target.state_dict()[k]) for k, v in target.items()))
            self.assertFalse(model.actor_optimizer.state)
            self.assertEqual(model.actor_update_count, 0)
        self.assertTrue(any(not torch.equal(v, model.critic.state_dict()[k]) for k, v in critic.items()))
        self.assertTrue(all(int(s['step']) == 4 for s in model.critic_optimizer.state.values()))
        self.assertTrue(model.update().metrics['actor_updated'])
        self.assertEqual(model.actor_update_count, 1)
        self.assertTrue(any(not torch.equal(v, model.actor.state_dict()[k]) for k, v in actor.items()))
        self.assertTrue(all(int(s['step']) == 1 for s in model.actor_optimizer.state.values()))

    def test_checkpoint_resumes_remaining_warmup_only(self):
        model = self.algorithm()
        model.update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            model.save_checkpoint(path)
            restored = self.algorithm()
            restored.load_checkpoint(path)
            self.assertEqual(restored.critic_warmup_completed, 1)
            self.assertFalse(restored.update().metrics['actor_updated'])
            self.assertTrue(restored.update().metrics['actor_updated'])
            # A BC recovery after initial warmup must not restart it.
            restored.load_actor_initialization(path, reset_optimizer=True)
            self.assertEqual(restored.critic_warmup_completed, 2)
            with self.assertRaisesRegex(ValueError, 'saved critic_warmup'):
                self.algorithm(0).load_checkpoint(path)

    def test_disabled_and_insufficient_buffer(self):
        model = self.algorithm(0)
        self.assertTrue(model.update().metrics['actor_updated'])
        model = self.algorithm()
        model.clear_experience()
        model.update()
        self.assertEqual(model.critic_warmup_completed, 0)

    def test_historical_configs_default_to_no_warmup(self):
        from configs.project_config import CONFIG
        from configs.runtime import validate
        old = copy.deepcopy(CONFIG)
        old['training'].pop('critic_warmup_updates')
        self.assertEqual(validate(old)['training']['critic_warmup_updates'], 0)
        old['training']['critic_warmup_updates'] = -1
        with self.assertRaises(ValueError):
            validate(old)

    def test_eval_skips_old_warmup_files_using_their_own_snapshot(self):
        import json
        from configs.project_config import CONFIG
        from launch import checkpoints, experiment_root
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(CONFIG)
            config['experiment']['output_root'] = directory
            config['evaluation'].update(only_latest=False, selected_global_step=None, start_global_step=0)
            folder = experiment_root(config) / 'checkpoints/eval_candidates'
            folder.mkdir(parents=True)
            old = copy.deepcopy(config)
            old['training']['critic_warmup_updates'] = 10
            for step in (5, 10, 15):
                path = folder / f'ckpt_gs{step:06d}.pth'
                path.touch()
                path.with_suffix('.config.json').write_text(json.dumps({'config': old}))
            self.assertEqual([p.name for p in checkpoints(config)], ['ckpt_gs000015.pth'])
            # New explicit counters take precedence over checkpoint numbering.
            path = folder / 'ckpt_gs000005.config.json'
            path.write_text(json.dumps({'config': old, 'checkpoint_training_state': {'actor_update_count': 1}}))
            self.assertEqual(len(checkpoints(config)), 2)
