"""Exploration distribution, reproducibility, contracts and legacy snapshots."""
import copy
import io
from pathlib import Path
import sys
import unittest
import torch
from exploration import GaussianActionNoise
from test_unified_env import FakeRuntime, NaoLocomotionEnv, ForwardWalkTaskV1, ForwardWalkTaskConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.project_config import CONFIG
from configs.runtime import validate


class ExplorationTests(unittest.TestCase):
    def test_zero_and_independent_rng(self):
        action = torch.zeros(10)
        rng = torch.get_rng_state().clone()
        self.assertIs(GaussianActionNoise(0, 1).apply(action), action)
        a, b = GaussianActionNoise(.01, 1), GaussianActionNoise(.01, 1)
        self.assertTrue(torch.equal(a.apply(action), b.apply(action)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.equal(a.apply(action), GaussianActionNoise(.01, 2).apply(action)))

    def test_distribution_and_checkpoint_continuation(self):
        noise = GaussianActionNoise(.01, 1)
        samples = noise.apply(torch.zeros(100000))
        self.assertLess(abs(samples.mean().item()), .0002)
        self.assertAlmostEqual(samples.std().item(), .01, delta=.0002)
        buffer = io.BytesIO()
        torch.save(noise.state_dict(), buffer)
        buffer.seek(0)
        restored = GaussianActionNoise(.01, 1)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(noise.apply(torch.zeros(10)), restored.apply(torch.zeros(10))))
        with self.assertRaises(ValueError):
            GaussianActionNoise(.02, 1).load_state_dict(noise.state_dict())
        with self.assertRaises(ValueError):
            restored.load_state_dict(None)

    def test_clipped_noisy_action_is_stored(self):
        runtime = FakeRuntime()
        env = NaoLocomotionEnv(runtime, ForwardWalkTaskV1(ForwardWalkTaskConfig()))
        env.initialize()
        env.reset()
        noisy = GaussianActionNoise(.01, 1).apply(torch.ones(10) * 20)
        step = env.step(noisy, 0)
        transition = env.finalize_transition(step, torch.zeros(10), rollout_fell=False,
                                             rollout_length=1, step_index=0)
        self.assertEqual(transition.action, runtime.adapter.adapt(noisy.tolist()))
        self.assertNotEqual(tuple(noisy.tolist()), transition.action.values)

    def test_legacy_defaults_to_disabled_and_rejects_invalid_std(self):
        old = copy.deepcopy(CONFIG)
        del old["training"]["action_noise_std_rad"]
        self.assertEqual(validate(old)["training"]["action_noise_std_rad"], 0)
        self.assertNotIn("action_noise_std_rad", old["training"])
        for value in (-1, float("nan"), float("inf")):
            config = copy.deepcopy(CONFIG)
            config["training"]["action_noise_std_rad"] = value
            with self.assertRaises(ValueError):
                validate(config)

if __name__ == "__main__":
    unittest.main()
