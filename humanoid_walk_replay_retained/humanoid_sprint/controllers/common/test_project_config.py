"""Configuration routing, immutable preparation and evaluation provenance tests."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from configs.project_config import CONFIG
from configs.runtime import (SNAPSHOT_ENV, algorithm_settings, load_config, require_launch,
                             runtime_settings, task_settings, validate)
from launch import experiment_root, prepare_run, render_world, resolve_config
from webots_platform.webots_runtime import WebotsRuntime


class ProjectConfigTests(unittest.TestCase):
    def config(self, folder):
        config = copy.deepcopy(CONFIG)
        config["initialization"]["resume"] = "auto"
        config["training"]["actor_lr"] = 1e-5
        config["experiment"]["output_root"] = str(folder)
        config["initialization"]["use_bc"] = False
        config["initialization"]["reset_to_bc_on_failure"] = False
        return config

    def test_preserves_original_training_values(self):
        config = copy.deepcopy(CONFIG)
        settings = algorithm_settings(config, "cpu", "forward_walk_v1")
        self.assertEqual((settings["actor_lr"], settings["critic_lr"], settings["batch_size"], settings["buffer_size"]), tuple(config["training"][key] for key in ("actor_lr", "critic_lr", "batch_size", "buffer_size")))
        self.assertEqual(task_settings(config, "train")["stop_x"], config['training']['stop_x_m'])
        self.assertEqual(task_settings(config, "eval")["stop_x"], config['evaluation']['stop_x_m'])
        self.assertEqual(task_settings(config, "eval")["sequence_length"], 35)
        self.assertEqual(config["training"]["update_every_steps"], 350)

    def test_rejects_unknown_and_inconsistent_parameters(self):
        for group, key, value in [("training", "min_buffer_size", 3), ("webots", "control_time_step_ms", 35),
                                  ("training", "buffer_size", 64), ("evaluation", "episodes", 0),
                                  ("environment", "settle_steps", -1), ("training", "actor_lr", float("nan"))]:
            config = copy.deepcopy(CONFIG)
            config[group][key] = value
            with self.assertRaises(ValueError):
                validate(config)

    def test_world_generation_changes_only_requested_fields(self):
        config = copy.deepcopy(CONFIG)
        template = PROJECT / config["webots"]["world_template"]
        before = template.read_bytes()
        config["webots"]["basic_time_step_ms"] = 10
        config["webots"]["initial_translation"] = [1, 2, 3]
        world = render_world(template, config)
        self.assertIn("basicTimeStep 10", world)
        self.assertIn("translation 1 2 3", world)
        self.assertIn('controller "configured_run"', world)
        self.assertIn("hidden rotation_22", world)
        self.assertEqual(template.read_bytes(), before)

    def test_prepared_snapshot_survives_current_config_change(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            snapshot, world, payload = prepare_run(config, "train")
            config["training"]["actor_lr"] = .5
            with patch.dict(os.environ, {SNAPSHOT_ENV: str(snapshot)}):
                self.assertEqual(load_config("train")["training"]["actor_lr"], 1e-5)
                require_launch()
                with self.assertRaises(ValueError):
                    load_config("eval")
            self.assertTrue(world.is_file())
            ini = world.parent.parent / "controllers/configured_run/runtime.ini"
            self.assertIn(CONFIG["webots"]["python"].replace("/", os.sep), ini.read_text())
            compile((ini.parent / "configured_run.py").read_text(), "wrapper", "exec")

    def test_eval_inherits_checkpoint_environment_and_keeps_eval_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config['training']['critic_warmup_updates'] = 0
            _, _, payload = prepare_run(config, "train")
            checkpoint = experiment_root(config) / "checkpoints/eval_candidates/ckpt_gs000005.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test model")
            checkpoint.with_suffix(".config.json").write_text(json.dumps(payload))
            config["environment"]["phase_coefficient"] = 99
            config["webots"]["control_time_step_ms"] = 80
            config["evaluation"]["stop_x_m"] = 4.4
            resolved, _, warnings = resolve_config(config, "eval")
            self.assertEqual(resolved["environment"]["phase_coefficient"], 4)
            self.assertEqual(resolved["webots"]["control_time_step_ms"], 40)
            self.assertEqual(resolved["evaluation"]["stop_x_m"], 4.4)
            self.assertFalse(warnings)

    def test_long_experiment_keeps_webots_resources_in_short_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config['experiment']['name'] = 'walk_critic_warmup_lr6_goal5p0_bcreset10001'
            destination = (experiment_root(config) / 'runs' /
                           'eval_batch_20260913T022642_708283Z_16b2fa' / 'gs000025')
            snapshot, world, _ = prepare_run(config, 'train', destination=destination)
            self.assertEqual(snapshot.parent, destination)
            self.assertEqual(world.parents[2], PROJECT / '.webots_runs')
            for source in (PROJECT / 'plugins').rglob('*'):
                if source.is_file():
                    copied = world.parent.parent / 'plugins' / source.relative_to(PROJECT / 'plugins')
                    self.assertLess(len(str(copied)), 260)
                    self.assertEqual(copied.read_bytes(), source.read_bytes())

    def test_legacy_eval_requires_policy_and_emits_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            checkpoint = experiment_root(config) / "checkpoints/eval_candidates/ckpt_gs000005.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"legacy")
            _, _, warnings = resolve_config(config, "eval")
            self.assertTrue(warnings)
            config["evaluation"]["allow_legacy_without_snapshot"] = False
            with self.assertRaises(ValueError):
                resolve_config(config, "eval")

    def test_resume_never_does_not_overwrite_existing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            checkpoint = experiment_root(config) / "checkpoints/resume.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"model")
            config["initialization"]["resume"] = "never"
            with self.assertRaises(ValueError):
                resolve_config(config, "train")

    def test_environment_parameters_reach_runtime(self):
        config = copy.deepcopy(CONFIG)
        config["environment"].update(sensor_warmup_steps=7, shoulder_position_rad=.9, motor_velocity_scale=.6, joint_limit_margin_rad=.002)
        runtime = WebotsRuntime(object(), **runtime_settings(config))
        self.assertEqual((runtime.sensor_warmup_steps, runtime.shoulder_position, runtime.motor_velocity_scale, runtime.joint_limit_margin), (7, .9, .6, .002))

    def test_saved_checkpoint_retains_its_run_configuration(self):
        from algorithms import load_algorithm
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            snapshot, _, _ = prepare_run(config, "train")
            algorithm = load_algorithm(config["training"]["algorithm_module"], config["training"]["algorithm_class"], algorithm_settings(config, "cpu", "forward_walk_v1"))
            checkpoint = Path(directory) / "saved.pth"
            with patch.dict(os.environ, {SNAPSHOT_ENV: str(snapshot)}):
                algorithm.save_checkpoint(checkpoint)
            saved = json.loads(checkpoint.with_suffix(".config.json").read_text())
            self.assertEqual(saved["config"]["environment"], config["environment"])
            self.assertEqual(saved["config"]["experiment"]["baseline_id"], config['experiment']['baseline_id'])
            algorithm.load_checkpoint(checkpoint)

    def test_prepared_run_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            resume = experiment_root(config) / "checkpoints/resume.pth"
            resume.parent.mkdir(parents=True)
            resume.write_bytes(b"before")
            snapshot, _, _ = prepare_run(config, "train")
            resume.write_bytes(b"after")
            with patch.dict(os.environ, {SNAPSHOT_ENV: str(snapshot)}):
                with self.assertRaises(RuntimeError):
                    require_launch()


if __name__ == "__main__":
    unittest.main()
