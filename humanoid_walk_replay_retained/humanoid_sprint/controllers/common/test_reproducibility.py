"""Network initialization and shared-Webots evaluation regression checks."""
import ast
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from configs.project_config import CONFIG
from configs.runtime import algorithm_settings
from algorithms import load_algorithm
from launch import experiment_root, prepare_evaluation, run_evaluation_batch


class ReproducibilityTests(unittest.TestCase):
    def test_sprinter_seeds_before_real_network_construction(self):
        # Execute the actual constructor with a stand-in Supervisor, avoiding Webots.
        tree = ast.parse((PROJECT / "controllers/sprinter/sprinter.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sprinter")
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("__init__", "set_global_seed")]
        namespace = dict(Supervisor=object, SEED=1, cfg=SimpleNamespace(DEVICE="cpu"),
                         torch=torch, np=np, random=random, RUN_CONFIG=CONFIG,
                         runtime_settings=lambda config, **kwargs: {}, WebotsRuntime=lambda *a, **k: object(),
                         TASK_MODULE="", TASK_CLASS="", task_config=lambda: {},
                         load_task=lambda *a: SimpleNamespace(task_id="forward_walk_v1"),
                         NaoLocomotionEnv=lambda *a: object(), load_algorithm=load_algorithm,
                         ALGORITHM_MODULE=CONFIG["training"]["algorithm_module"], ALGORITHM_CLASS=CONFIG["training"]["algorithm_class"],
                         algorithm_config=lambda device, task_id: algorithm_settings(CONFIG, device, task_id))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "sprinter_constructor", "exec"), namespace)
        def initial(seed):
            controller = namespace["Sprinter"](seed=seed)
            return {name: {k: v.clone() for k, v in getattr(controller.algorithm, name).state_dict().items()}
                    for name in ("actor", "critic")}, (random.random(), np.random.rand(), torch.rand(1).item())
        first, rng = initial(7)
        torch.rand(99); random.random(); np.random.rand(99)
        second, rng2 = initial(7)
        third, _ = initial(8)
        self.assertEqual(rng, rng2)
        for name in first:
            self.assertTrue(all(torch.equal(first[name][k], second[name][k]) for k in first[name]))
            self.assertTrue(any(not torch.equal(first[name][k], third[name][k]) for k in first[name]))

    def prepare(self, directory):
        config = copy.deepcopy(CONFIG)
        config["experiment"]["output_root"] = directory
        config["evaluation"].update(only_latest=False, selected_global_step=None, start_global_step=0,
                                    allow_legacy_without_snapshot=True)
        folder = experiment_root(config) / "checkpoints/eval_candidates"
        folder.mkdir(parents=True)
        for step in (5, 10):
            (folder / f"ckpt_gs{step:06d}.pth").write_bytes(str(step).encode())
        return prepare_evaluation(config)

    def test_one_webots_process_and_verified_world_loads(self):
        from evaluation_batch import finish_evaluation
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory)
            loads, quits = [], []
            controller = SimpleNamespace(timeStep=40, worldLoad=loads.append,
                                         step=lambda dt: -1, simulationQuit=quits.append)
            def wait():
                for run in json.loads(path.read_text())["runs"]:
                    payload = json.loads(Path(run["snapshot"]).read_text())
                    self.assertEqual(payload["config"]["evaluation"]["episodes"], 1)
                    self.assertEqual(len(payload["sources"]), 1)
                    folder = Path(run["snapshot"]).parent
                    (folder / "evaluation_result.json").write_text(json.dumps(dict(
                        eval_run_id=payload["run_id"], checkpoint_sha256=run["sha256"],
                        reach_time_mean=113.84, lateral_mean=.54)))
                    finish_evaluation(controller, payload)
                return 0
            with patch("launch.subprocess.Popen", return_value=SimpleNamespace(pid=123, wait=wait)) as process:
                run_evaluation_batch(path)
            self.assertEqual(process.call_count, 1)
            self.assertEqual(len(loads), 1)
            self.assertEqual(quits, [0])
            self.assertEqual(json.loads(path.read_text())["status"], "completed")

    def test_closed_or_failed_controller_stops_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory)
            with patch("launch.subprocess.Popen", return_value=SimpleNamespace(pid=123, wait=lambda: 0)) as process:
                with self.assertRaises(RuntimeError):
                    run_evaluation_batch(path)
            self.assertEqual(process.call_count, 1)
            self.assertEqual(json.loads(path.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
