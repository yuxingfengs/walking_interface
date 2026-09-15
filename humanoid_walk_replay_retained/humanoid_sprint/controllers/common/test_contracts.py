"""Regression tests for the S1 shared data contracts."""

import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from checkpointing import load_actor_weights
from contracts import (
    ACTION_DIM,
    ACTION_JOINT_ORDER,
    ACTION_SPEC,
    ActorObservation,
    CanonicalAction,
    OBSERVATION_FIELDS,
    OBSERVATION_SPEC,
    STATE_DIM,
    Transition,
    contract_metadata,
    validate_contract_metadata,
)
from models import BoundedActor


class ContractTests(unittest.TestCase):
    def test_v1_dimensions_and_orders(self):
        self.assertEqual(STATE_DIM, 18)
        self.assertEqual(ACTION_DIM, 10)
        self.assertEqual(OBSERVATION_SPEC.fields, OBSERVATION_FIELDS)
        self.assertEqual(ACTION_SPEC.joint_order, ACTION_JOINT_ORDER)

    def test_transition_keeps_boundary_causes_separate(self):
        observation = ActorObservation((0.0,) * STATE_DIM)
        action = CanonicalAction((0.0,) * ACTION_DIM)
        transition = Transition(observation, action, 1.0, observation, False, True)
        self.assertFalse(transition.terminated)
        self.assertTrue(transition.truncated)
        self.assertTrue(transition.done)

    def test_incompatible_contract_is_rejected(self):
        metadata = contract_metadata()
        metadata["action_id"] = "different_action"
        with self.assertRaises(ValueError):
            validate_contract_metadata(metadata, artifact="test")

    def test_new_and_legacy_actor_artifacts_load(self):
        source = BoundedActor(STATE_DIM, 256, ACTION_DIM)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            current_path = directory / "current.pth"
            legacy_path = directory / "legacy.pth"
            torch.save({
                "actor_state_dict": source.state_dict(),
                "meta": contract_metadata(),
            }, current_path)
            torch.save(source.state_dict(), legacy_path)

            load_actor_weights(
                BoundedActor(STATE_DIM, 256, ACTION_DIM), current_path, torch.device("cpu")
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_actor_weights(
                    BoundedActor(STATE_DIM, 256, ACTION_DIM), legacy_path, torch.device("cpu")
                )
            self.assertTrue(any("legacy" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
