"""Regression tests for the standardized S3 BC artifact pipeline."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from bc_trainer import BCTrainer, BCTrainingConfig
from checkpointing import load_actor_weights, load_torch_artifact
from contracts import ACTION_DIM, STATE_DIM, contract_metadata
from demonstration_dataset import DemonstrationDataset
from models import BoundedActor


class BCPipelineTests(unittest.TestCase):
    def write_dataset(self, directory: Path, *, action_id=None) -> Path:
        path = directory / "demo.csv"
        metadata = contract_metadata()
        if action_id is not None:
            metadata["action_id"] = action_id
        fieldnames = (
            [
                "demo_name", "expert_id", "uses_privileged_state",
                "task_id", "observation_id", "action_id",
            ]
            + [f"state_{index}" for index in range(STATE_DIM)]
            + [f"action_{index}" for index in range(ACTION_DIM)]
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row_index in range(4):
                writer.writerow({
                    "demo_name": "test",
                    "expert_id": "test_expert",
                    "uses_privileged_state": False,
                    **metadata,
                    **{
                        f"state_{index}": float(row_index + index) / 100.0
                        for index in range(STATE_DIM)
                    },
                    **{
                        f"action_{index}": float(index) / 100.0
                        for index in range(ACTION_DIM)
                    },
                })
        path.with_suffix(".metadata.json").write_text(json.dumps({
            **metadata,
            "expert_id": "test_expert",
            "uses_privileged_state": False,
        }), encoding="utf-8")
        return path

    def test_versioned_dataset_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = DemonstrationDataset.load(self.write_dataset(Path(directory)))
            self.assertEqual(dataset.states.shape, (4, STATE_DIM))
            self.assertEqual(dataset.actions.shape, (4, ACTION_DIM))
            self.assertEqual(dataset.provenance.samples, 4)
            self.assertEqual(len(dataset.provenance.dataset_sha256), 64)
            self.assertFalse(dataset.provenance.is_legacy)

    def test_incompatible_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(Path(directory), action_id="wrong_action")
            with self.assertRaises(ValueError):
                DemonstrationDataset.load(path)

    def test_trainer_exports_reproducible_loadable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dataset = DemonstrationDataset.load(self.write_dataset(directory))
            config = BCTrainingConfig(epochs=1, batch_size=2, learning_rate=1e-4, seed=7)
            first = BCTrainer(config, torch.device("cpu")).train(dataset, directory / "first.pth")
            second = BCTrainer(config, torch.device("cpu")).train(dataset, directory / "second.pth")
            first_artifact = load_torch_artifact(first.artifact_path, "cpu")
            second_artifact = load_torch_artifact(second.artifact_path, "cpu")
            self.assertEqual(first.metadata["artifact_type"], "bc_actor")
            self.assertEqual(first.metadata["dataset_sha256"], dataset.provenance.dataset_sha256)
            for key, value in first_artifact["actor_state_dict"].items():
                self.assertTrue(torch.equal(value, second_artifact["actor_state_dict"][key]))
            load_actor_weights(
                BoundedActor(STATE_DIM, 256, ACTION_DIM),
                first.artifact_path,
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
