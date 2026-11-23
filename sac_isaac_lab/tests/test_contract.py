"""Regression checks for SAC training configuration, dispatch and artifact provenance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runner'))
from configuration import CODE, SAC, SIM, launch_configuration, load_configuration, read_json, sha


class SingleRobotContractTests(unittest.TestCase):
    def test_local_launch_settings(self):
        opts = dict(current_experiment='walk_lateral_random_vx', num_envs={'walk_lateral_random_vx': 2},
                    sim_device='cpu', learner_device='cuda:0', headless=True)
        settings = launch_configuration(options=opts)['isaac']
        self.assertEqual((settings['num_envs'], settings['device'], settings['learner_device']), (2, 'cpu', 'cuda:0'))
        self.assertEqual(launch_configuration('omni', opts)['isaac']['num_envs'], 1)
        with self.assertRaisesRegex(ValueError, 'single-robot'):
            launch_configuration('omni', dict(opts, num_envs=2))
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            launch_configuration(options=dict(opts, num_envs=False))

    def test_random_walk_offline_check_reports_launch_settings(self):
        from offline_check import check
        options = dict(current_experiment='walk_lateral_random_vx', num_envs=4,
                       sim_device='cuda:0', learner_device='cpu')
        report = check(cfg=launch_configuration(options=options))
        self.assertEqual((report['num_envs'], report['sim_device'], report['learner_device']),
                         (4, 'cuda:0', 'cpu'))
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertGreater(report['tracking_scale_m_s'], 0)

    def test_walk_only_body_commands_and_reward(self):
        import numpy as np
        cfg = load_configuration(experiment='walk_only')
        sys.path.insert(0, str(CODE/'training'))
        from basewalk.fixed_cycle import online_stages
        from basewalk.reward_dispatch import implementation
        reward = implementation(cfg)
        stages = online_stages(cfg)
        spec = cfg['training']['walk_commands']
        for xy in spec['commands_m_s']:
            for _ in range(spec['hold_steps']):
                np.testing.assert_allclose(stages.command(0, 0), [*xy, 0.])
                stages.advance_phase(.9, .1, fell=True)
        np.testing.assert_allclose(stages.command(0, 0), online_stages(cfg).command(0, 0))
        command = np.array([.065, .03, 0.])
        def score(velocity, target=command):
            return reward.motion_reward(target, velocity, [0., 0.], [0., 0., -1.], 2, cfg)[0]
        self.assertAlmostEqual(score(command), 2.5)
        self.assertLess(score([.079, 0., 0.]), score([.065, 0., 0.]))
        self.assertAlmostEqual(score([.26, .03, 0.]), 1.25 * (2. - .195 / .08))
        self.assertAlmostEqual(score([.065, 0., 0.]), 1.25 * (2. - .03 / .08))
        self.assertAlmostEqual(score([.035, .03, 0.]), 1.25 * (2. - .03 / .08))
        self.assertAlmostEqual(score([.08, .04, 0.], [.08, 0., 0.]), 1.875)
        self.assertAlmostEqual(score([0., 0., 0.], [.08, 0., 0.]), 1.25)
        self.assertAlmostEqual(score([.04, -.015, 0.], [0., -.03, 0.]), 1.25 * (2. - .055 / .08))
        self.assertLess(score([.065, .03, .11]), score(command))
        obs = np.zeros((1, 55))
        obs[0, 5:8] = command
        nxt = obs.copy()
        nxt[0, 47:49] = [.065, 0.]
        nxt[0, 50] = .11 * cfg['environment']['yaw_rate_scale']
        np.testing.assert_allclose(reward.revalue_arrays(dict(obs=obs, next_obs=nxt, modes=[2]), cfg),
                                   [score([.065, 0., .11])])

    def test_004_preserves_source_training_and_uses_standing_reset(self):
        cfg = load_configuration(experiment='004')
        folder = SAC/'experiments/004'
        source = read_json(folder/'training_config.json')
        common = read_json(CODE/'training_config.json')
        self.assertEqual(cfg['training'], {**source['training'], **common['training']})
        self.assertEqual(cfg['sac'], common['sac'])
        self.assertEqual(cfg['reward_source_sha256'], sha(folder/'reward.py'))
        self.assertEqual(cfg['initialization']['sha256'], read_json(folder/'bc/manifest.json')['checkpoint_sha256'])
        self.assertEqual(Path(cfg['project_layout']['runs_root']), CODE/'experiments/training_trials/004/runs')
        self.assertEqual(cfg['environment']['settle_steps'], source['environment']['settle_steps'])
        settings = cfg['isaac']
        pose = read_json(SIM/'assets/ddpg_standing_pose.json')
        self.assertEqual(settings['standing_joint_positions'], pose['joint_positions_rad'])
        self.assertEqual(settings['spawn_height'], pose['settled_height_above_floor_m'])
        self.assertEqual((settings['solver_position_iterations'], settings['solver_velocity_iterations']), (48, 8))

    def test_budget_does_not_change_webots_source(self):
        source = SAC / 'experiments/001/training_config.json'
        before = source.read_bytes()
        cfg = load_configuration(experiment='003')
        shared = read_json(SAC / 'config.json')
        self.assertEqual(cfg['training']['total_env_steps'],
                         shared['training']['total_env_steps'])
        self.assertEqual(cfg['budget_source']['path'], str(SAC/'config.json'))
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(cfg['environment']['floor_z'], 0.)
        self.assertNotIn('webots', cfg)

    def test_reject_parallel_or_changed_control_interval(self):
        settings = read_json(SIM / 'setup/simulator_config.json')
        with tempfile.TemporaryDirectory() as location:
            path = Path(location) / 'config.json'
            for key, value in [('num_envs', 2), ('physics_dt', .01)]:
                path.write_text(json.dumps(dict(settings, **{key: value})), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_configuration(path)

    def test_snapshot_owns_bc_and_reward_bytes(self):
        from main import snapshot
        with tempfile.TemporaryDirectory() as location:
            folder = Path(location)
            asset = folder / 'hash_fixture.bin'
            asset.write_bytes(b'provenance fixture, not a physics model')
            cfg = snapshot(folder, load_configuration()['isaac'], asset)
            self.assertEqual(sha(cfg['initialization']['actor_path']), cfg['initialization']['sha256'])
            self.assertEqual(sha(cfg['reward_module_path']), cfg['reward_source_sha256'])
            self.assertEqual(read_json(folder / 'effective_config.json')['budget_source'], cfg['budget_source'])
            self.assertTrue((folder / 'source/bc_manifest.json').is_file())
            self.assertEqual(sha(folder/'source/project_config.json'), sha(SAC/'config.json'))
            self.assertIn('basewalk/sac.py', cfg['source_hashes'])
            for name in ('environment.py', 'batched_environment.py', 'joint_initialization.py'):
                self.assertEqual(cfg['source_hashes'][name], sha(SIM/name))
                self.assertEqual(sha(folder/'source'/name), sha(SIM/name))
            sys.path.remove(str(folder / 'source'))

    def test_public_config_is_the_only_external_settings_file(self):
        from unittest.mock import patch
        import configuration
        original = configuration.read_json
        seen = []
        def read_local(path):
            path = Path(path).resolve()
            self.assertTrue(SAC.resolve() in path.parents or path in {
                CODE/'config.json', CODE/'training_config.json', SIM/'setup/simulator_config.json', SIM/'assets/ddpg_standing_pose.json'})
            seen.append(path)
            data = original(path)
            if path == (SAC/'config.json').resolve():
                data['training']['total_env_steps'] = 12345
            return data
        with patch.object(configuration, 'read_json', side_effect=read_local):
            cfg = load_configuration(experiment='003')
        self.assertEqual(cfg['training']['total_env_steps'], 12345)
        self.assertIn((SAC/'config.json').resolve(), seen)

    def test_falls_always_count_without_public_option(self):
        self.assertNotIn('online_count_fall_as_attempt', read_json(SAC/'config.json')['training'])
        cfg = load_configuration()
        self.assertNotIn('online_count_fall_as_attempt', cfg['training'])
        sys.path.insert(0, str(CODE/'training'))
        from basewalk.fixed_cycle import FixedCycleStages
        stages = FixedCycleStages(cfg, cycle_limit=20,
                                  turn_only=True, fixed_turn_sign=1.)
        stages.advance_phase(.2, .3, fell=True)
        self.assertEqual(stages.cycles, 1)
        stages.advance_phase(.9, .1, fell=True)
        self.assertEqual(stages.cycles, 2)  # fall + phase wrap counts only once
        self.assertFalse(hasattr(stages, 'count_fall_as_attempt'))
        stages.advance_phase(.2, .3)
        self.assertEqual(stages.cycles, 2)
        stages.advance_phase(.9, .1)
        self.assertEqual(stages.cycles, 3)


if __name__ == '__main__':
    unittest.main()
