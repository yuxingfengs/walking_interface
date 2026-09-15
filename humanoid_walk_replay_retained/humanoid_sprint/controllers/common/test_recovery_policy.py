import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from recovery_policy import within_bc_reset_window


class RecoveryPolicyTests(unittest.TestCase):
    def test_cutoff_and_historical_default(self):
        config = {'bc_reset_max_rollouts': 1000}
        self.assertTrue(within_bc_reset_window(config, 999))
        self.assertTrue(within_bc_reset_window(config, 1000))
        self.assertFalse(within_bc_reset_window(config, 1001))
        # Resume after 1000 completed rollouts starts at rollout 1001.
        self.assertFalse(within_bc_reset_window(config, 1000 + 1))
        self.assertTrue(within_bc_reset_window({}, 10000))

    def test_actual_recovery_branch_preserves_learning_after_cutoff(self):
        source = Path(__file__).resolve().parents[1] / 'sprinter/sprinter.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.BoolOp) and len(n.test.values) == 3
                      and isinstance(n.test.values[-1], ast.Name)
                      and n.test.values[-1].id == 'bc_reset_allowed')
        code = compile(ast.Module(body=[branch], type_ignores=[]), str(source), 'exec')
        for fell in (False, True):
            actor, replay = object(), object()
            algorithm = SimpleNamespace(actor=actor, replay_buffer=replay,
                load_actor_initialization=Mock(side_effect=AssertionError('Actor must not reset')),
                clear_experience=Mock(side_effect=AssertionError('Replay must not clear')))
            reset = Mock()
            host = SimpleNamespace(algorithm=algorithm, env=SimpleNamespace(reset=reset))
            namespace = dict(self=host, bc_reset_reason='consecutive_stall_cycles', bc_reset_allowed=False,
                RUN_CONFIG={'initialization': {'reset_to_bc_on_failure': True}, 'environment': {'settle_steps': 64}},
                fell=fell, completed_rollouts=1001, INITIAL_ACTIONS=[torch.zeros(10)], torch=torch,
                consecutive_fall_cycles=15, consecutive_stall_cycles=40, episodes_since_train=7)
            exec(code, namespace)
            self.assertIs(host.algorithm.actor, actor)
            self.assertIs(host.algorithm.replay_buffer, replay)
            self.assertEqual(reset.call_count, 0 if fell else 1)
            self.assertEqual(namespace['episodes_since_train'], 7)
            self.assertEqual(namespace['consecutive_stall_cycles'], 0)

    def test_config_compatibility(self):
        from configs.runtime import validate
        from configs.project_config import CONFIG
        config = copy.deepcopy(CONFIG)
        config['initialization'].pop('bc_reset_max_rollouts', None)
        self.assertEqual(validate(config)['initialization']['bc_reset_max_rollouts'], 0)
        config['initialization']['bc_reset_max_rollouts'] = -1
        with self.assertRaises(ValueError):
            validate(config)
