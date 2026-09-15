"""Behavioral tests: logging must not influence collection, rewards or learning."""
import unittest
from types import SimpleNamespace
from continuous_training import ContinuousCollector
from contracts import ActorObservation, CanonicalAction
from envs.nao_locomotion_env import NaoLocomotionEnv, EnvStep
from tasks.forward_walk_v1 import ForwardWalkTaskV1, ForwardWalkTaskConfig


class FakeEnv:
    step_transition = NaoLocomotionEnv.step_transition

    def __init__(self, falls=(), goals=(), moving=True):
        self.task = ForwardWalkTaskV1(ForwardWalkTaskConfig())
        self.falls, self.goals = falls, goals
        self.moving = moving
        self.actions = 0
        self.resets = 0
        self.position = [0, 0, .7]

    def reset(self, settle_steps):
        self.resets += 1
        self.position[0] = 0
        self.obs = ActorObservation(tuple([0.] * 18))
        return self.obs

    def step(self, action, index):
        self.actions += 1
        self.position[0] += .01 if self.moving else 0
        successor = ActorObservation(tuple([self.position[0]] + [0.] * 17))
        s = EnvStep(self.obs, CanonicalAction(tuple(action)), successor,
                    .01 if self.moving else 0, self.actions in self.falls, False,
                    dict(fell=self.actions in self.falls, reached_goal=self.actions in self.goals,
                         position=tuple(self.position), estimated_forward=self.position[0], odometry_valid=True))
        self.obs = successor
        return s


class FakeAlgorithm:
    batch_size = 2
    def __init__(self):
        self.replay = []
        self.all_transitions = []
        self.updates = []
        self.bc_loads = 0
    @property
    def experience_size(self):
        return len(self.replay)
    def set_evaluation_mode(self):
        pass
    def act(self, observation):
        return [.1] * 10
    def observe(self, transition):
        self.replay.append(transition)
        self.all_transitions.append(transition)
    def update(self):
        self.updates.append(len(self.all_transitions))
        return SimpleNamespace(metrics={})
    def load_actor_initialization(self, *args, **kwargs):
        self.bc_loads += 1
        return True
    def clear_experience(self):
        self.replay.clear()


class Noise:
    def apply(self, action):
        return action
    def state_dict(self):
        return {}


class ContinuousTests(unittest.TestCase):
    def test_noise_schedule_boundary_and_resume_preserve_rng(self):
        import torch
        from exploration import GaussianActionNoise
        algorithm = FakeAlgorithm()
        algorithm.act = lambda obs: torch.zeros(10)
        training = dict(noise_switch_env_steps=70000, action_noise_std_rad=.002,
                        action_noise_std_after_switch=.001)
        init = dict(stall_window_steps=2)
        noise = GaussianActionNoise(.002, 1)
        collector = ContinuousCollector(FakeEnv(), algorithm, noise, training, init)
        collector.env_steps = 69999
        saved = collector.metadata()
        first = collector.exploration_action(None)
        mirror = GaussianActionNoise(.002, 1)
        mirror.load_state_dict(saved['exploration'])
        self.assertTrue(torch.equal(first, mirror.apply(torch.zeros(10), std_rad=.002)))
        collector.env_steps = 70000
        saved = collector.metadata()
        second = collector.exploration_action(None)
        restored = GaussianActionNoise(.002, 1)
        restored.load_state_dict(saved['exploration'])
        resumed = ContinuousCollector(FakeEnv(), algorithm, restored, training, init, metadata=saved)
        self.assertTrue(torch.equal(second, resumed.exploration_action(None)))
        mirror.load_state_dict(saved['exploration'])
        self.assertTrue(torch.equal(second, mirror.apply(torch.zeros(10), std_rad=.001)))

    def test_fast_warmup_respects_buffer_then_switches_to_joint_interval(self):
        algorithm = FakeAlgorithm()
        algorithm.critic_warmup_updates = 3
        algorithm.critic_warmup_completed = 0
        original_update = algorithm.update
        def update():
            result = original_update()
            if algorithm.critic_warmup_completed < algorithm.critic_warmup_updates:
                algorithm.critic_warmup_completed += 1
            return result
        algorithm.update = update
        training = dict(total_env_steps=20, update_every_steps=5,
                        critic_warmup_update_every_steps=2, log_every_steps=3,
                        min_buffer_for_train=6, actor_lr=1e-5)
        initialization = dict(stall_window_steps=2, stall_timeout_steps=100,
                              stall_min_forward_progress_m=.001, consecutive_falls=15,
                              bc_reset_max_env_steps=0, reset_to_bc_on_failure=False)
        collector = ContinuousCollector(FakeEnv(goals=(7, 12)), algorithm, Noise(),
                                        training, initialization)
        collector.run()
        self.assertEqual(algorithm.updates, [6, 8, 10, 15, 20])
        self.assertEqual(algorithm.critic_warmup_completed, 3)

    def test_lr_stage_boundary_and_resume_preserve_optimizer(self):
        algorithm = FakeAlgorithm()
        algorithm.actor_optimizer = SimpleNamespace(param_groups=[{'lr': 9.0}], state={'moment': 42})
        algorithm.critic_optimizer = SimpleNamespace(param_groups=[{'lr': 9.0}], state={'moment': 24})
        t = dict(lr_switch_env_steps=35000, actor_lr=1e-5, critic_lr=1e-6,
                 actor_lr_after_switch=1e-6, critic_lr_after_switch=1e-6)
        c = ContinuousCollector(FakeEnv(), algorithm, Noise(), t, dict(stall_window_steps=35))
        original = algorithm.actor_optimizer
        for step, expected in [(34999, 1e-5), (35000, 1e-5), (35001, 1e-6), (70000, 1e-6)]:
            c.env_steps = step
            c.apply_learning_rate_stage()
            self.assertEqual(original.param_groups[0]['lr'], expected)
            self.assertEqual(algorithm.critic_optimizer.param_groups[0]['lr'], 1e-6)
        resumed = ContinuousCollector(FakeEnv(), algorithm, Noise(), t,
                                      dict(stall_window_steps=35), metadata=c.metadata())
        original.param_groups[0]['lr'] = 8.0
        resumed.apply_learning_rate_stage()
        self.assertEqual(original.param_groups[0]['lr'], 1e-6)
        self.assertIs(original, algorithm.actor_optimizer)
        self.assertEqual(original.state, {'moment': 42})

    def test_disabled_lr_schedule_preserves_historical_behavior(self):
        c = ContinuousCollector(FakeEnv(), FakeAlgorithm(), Noise(), {}, dict(stall_window_steps=35))
        c.env_steps = 70000
        c.apply_learning_rate_stage()  # Legacy algorithms need no optimizer access.

    def test_reset_maps_sensor_joints_to_action_order(self):
        from contracts import OBSERVATION_FIELDS, ACTION_JOINT_ORDER
        env = FakeEnv()
        values = tuple(float(i)/100 for i in range(18))
        env.reset = lambda settle_steps: ActorObservation(values)
        c = ContinuousCollector(env, FakeAlgorithm(), Noise(), {},
                                dict(stall_window_steps=35))
        _, previous = c.reset()
        self.assertEqual(previous, tuple(values[OBSERVATION_FIELDS.index(n)] for n in ACTION_JOINT_ORDER))
        self.assertNotEqual(previous, values[-10:])

    def run_collector(self, log=3, falls=(), goals=(), cutoff=0, recovery=False,
                      moving=True, budget=20, metadata=None):
        env, algorithm = FakeEnv(falls, goals, moving), FakeAlgorithm()
        t = dict(total_env_steps=budget, update_every_steps=5, log_every_steps=log,
                 min_buffer_for_train=2, actor_lr=1e-5)
        i = dict(stall_window_steps=2, stall_timeout_steps=3, stall_min_forward_progress_m=.001,
                 consecutive_falls=1, bc_reset_max_env_steps=cutoff,
                 reset_to_bc_on_failure=recovery, bc_checkpoint='test')
        events = []
        c = ContinuousCollector(env, algorithm, Noise(), t, i,
                                lambda kind, c, **v: events.append((kind, c.env_steps, v)), metadata)
        c.run()
        return c, env, algorithm, events

    def test_logging_is_observational_and_every_action_is_stored(self):
        a = self.run_collector(log=3)
        b = self.run_collector(log=7)
        self.assertEqual(a[2].all_transitions, b[2].all_transitions)
        self.assertEqual(a[2].updates, [5, 10, 15, 20])
        self.assertEqual(a[2].updates, b[2].updates)
        self.assertEqual(a[1].actions, 20)
        self.assertEqual(a[1].resets, 1)
        self.assertEqual(sum(v['seq_len'] for k, n, v in a[3] if k == 'window'), 20)

    def test_goal_keeps_replay_and_schedule_with_pre_reset_successor(self):
        c, env, algo, _ = self.run_collector(goals=(4, 9))
        self.assertEqual(algo.updates, [5, 10, 15, 20])
        self.assertEqual(len(algo.replay), 20)
        self.assertFalse(algo.replay[3].done)
        self.assertAlmostEqual(algo.replay[3].next_observation.values[0], .04)
        self.assertEqual(algo.replay[4].observation.values[0], 0)

    def test_fall_penalty_only_on_actual_fall(self):
        _, _, algo, _ = self.run_collector(falls=(4,))
        self.assertEqual([i+1 for i, r in enumerate(algo.replay) if r.reward < 0], [4])
        self.assertEqual(algo.replay[3].reward, -3)
        self.assertTrue(algo.replay[3].done)
        self.assertFalse(algo.replay[2].done)

    def test_bc_cutoff_inclusive_and_post_cutoff_preserves_buffer(self):
        c, env, algo, events = self.run_collector(falls=(5, 6), cutoff=5, recovery=True)
        self.assertEqual(algo.bc_loads, 1)
        self.assertEqual(len(algo.replay), 15)
        self.assertEqual(algo.updates, [10, 15, 20])
        self.assertEqual(c.env_steps, 20)

    def test_stall_recovers_independently_of_log_interval(self):
        a = self.run_collector(log=3, moving=False)
        b = self.run_collector(log=7, moving=False)
        self.assertEqual([(k,n) for k,n,v in a[3] if k=='reset'],
                         [(k,n) for k,n,v in b[3] if k=='reset'])
        self.assertGreater(a[1].resets, 1)

    def test_resume_uses_cumulative_budget_and_update_interval(self):
        c, _, _, _ = self.run_collector(budget=7)
        resumed, env, algo, _ = self.run_collector(budget=20, metadata=c.metadata())
        self.assertEqual(env.actions, 13)
        self.assertEqual(algo.updates, [3, 8, 13])
        self.assertEqual(resumed.env_steps, 20)
        self.assertEqual(resumed.update_count, 4)

    def test_stall_uses_odometry_even_if_world_position_moves(self):
        env, algorithm = FakeEnv(moving=True), FakeAlgorithm()
        original = env.step
        def step(action, index):
            result = original(action, index)
            result.info['estimated_forward'] = 0.
            result.info['position'] = (1000.*env.actions, 9000., .7)
            return result
        env.step = step
        t = dict(total_env_steps=10, update_every_steps=5, log_every_steps=3,
                 min_buffer_for_train=2, actor_lr=1e-5)
        i = dict(stall_window_steps=2, stall_timeout_steps=3, stall_min_forward_progress_m=.001,
                 consecutive_falls=5, bc_reset_max_env_steps=0,
                 reset_to_bc_on_failure=True, bc_checkpoint='test')
        events=[]
        collector=ContinuousCollector(env,algorithm,Noise(),t,i,
            lambda kind,c,**v:events.append((kind,c.env_steps,v)))
        collector.run()
        self.assertEqual([n for kind,n,v in events if kind=='reset'],[3,6,9])
        self.assertEqual(algorithm.bc_loads,3)

    def test_rejects_legacy_resume(self):
        with self.assertRaises(ValueError):
            self.run_collector(metadata={'global_step': 10})


if __name__ == '__main__':
    unittest.main()
