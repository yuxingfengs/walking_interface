"""Recovery and phase boundaries for the independent retained-Replay experiment."""
import copy
import unittest
from types import SimpleNamespace

import torch
from configs.runtime import load_config
from continuous_training import ContinuousCollector
from exploration import GaussianActionNoise
from test_continuous_training import FakeAlgorithm, FakeEnv


class RetainedReplayTests(unittest.TestCase):
    def run_case(self, falls=(), moving=True):
        config = load_config()
        t, init = copy.deepcopy(config['training']), copy.deepcopy(config['initialization'])
        t.update(total_env_steps=20, lr_switch_env_steps=10, noise_switch_env_steps=10,
                 update_every_steps=3, min_buffer_for_train=2, log_every_steps=5)
        init.update(consecutive_falls=1, stall_window_steps=2, stall_timeout_steps=3)
        env, alg = FakeEnv(falls=falls, moving=moving), FakeAlgorithm()
        alg.act = lambda obs: torch.zeros(10)
        alg.actor_optimizer = SimpleNamespace(param_groups=[{'lr': t['actor_lr']}])
        alg.critic_optimizer = SimpleNamespace(param_groups=[{'lr': t['critic_lr']}])
        alg.clear_experience = lambda: self.fail('Recovery must not clear Replay')
        recovered = []
        def recover(*args, **kwargs):
            recovered.append((env.actions, kwargs['actor_lr']))
            alg.actor_optimizer.param_groups[0]['lr'] = kwargs['actor_lr']
            return True
        alg.load_actor_initialization = recover
        actions, rates = [], []
        original_step = env.step
        def step(action, index):
            actions.append(action.clone())
            return original_step(action, index)
        env.step = step
        def emit(kind, collector, **kwargs):
            if kind == 'update':
                rates.append((collector.env_steps, alg.actor_optimizer.param_groups[0]['lr']))
        collector = ContinuousCollector(env, alg, GaussianActionNoise(.002, 1), t, init, emit)
        collector.run()
        return collector, alg, actions, rates, recovered

    def test_failure_transition_survives_recovery_and_halfway_switch(self):
        c, alg, actions, rates, recovered = self.run_case(falls=(10, 11))
        self.assertEqual(len(alg.replay), 20)
        self.assertTrue(alg.replay[9].done)
        self.assertTrue(alg.replay[10].done)
        self.assertEqual(recovered, [(10, 1e-5), (11, 1e-6)])
        self.assertTrue(all(torch.count_nonzero(a) > 0 for a in actions[:10]))
        self.assertTrue(all(torch.equal(a, torch.zeros(10)) for a in actions[10:]))
        self.assertTrue(all(lr == (1e-5 if step <= 10 else 1e-6) for step, lr in rates))
        self.assertEqual(c.env_steps, 20)

    def test_stall_recovery_also_preserves_all_transitions(self):
        _, alg, _, _, recovered = self.run_case(moving=False)
        self.assertGreater(len(recovered), 0)
        self.assertEqual(len(alg.replay), 20)

    def test_actual_config_and_resume_boundary(self):
        config = load_config(); t, init = config['training'], config['initialization']
        self.assertEqual(t['total_env_steps'], 200000)
        self.assertEqual(t['lr_switch_env_steps'], t['total_env_steps']//2)
        self.assertEqual(t['noise_switch_env_steps'], t['total_env_steps']//2)
        self.assertFalse(init['clear_replay_on_bc_reset'])
        alg = FakeAlgorithm(); alg.act = lambda obs: torch.zeros(10)
        alg.actor_optimizer = SimpleNamespace(param_groups=[{'lr': 1e-5}])
        alg.critic_optimizer = SimpleNamespace(param_groups=[{'lr': 1e-6}])
        noise = GaussianActionNoise(.002, 1)
        c = ContinuousCollector(FakeEnv(), alg, noise, t, init,
            metadata={'continuous_training': {'env_steps': 100000}})
        rng = noise.generator.get_state().clone()
        self.assertTrue(torch.equal(c.exploration_action(None), torch.zeros(10)))
        self.assertTrue(torch.equal(rng, noise.generator.get_state()))
        c.env_steps = 100001; c.apply_learning_rate_stage()
        self.assertEqual(alg.actor_optimizer.param_groups[0]['lr'], 1e-6)
        self.assertEqual(alg.critic_optimizer.param_groups[0]['lr'], 1e-6)


if __name__ == '__main__':
    unittest.main()
