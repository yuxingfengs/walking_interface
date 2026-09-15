"""Training-only Gaussian exploration with an independent, checkpointed RNG."""
import math
import torch


class GaussianActionNoise:
    def __init__(self, std_rad, seed):
        self.std_rad = float(std_rad)
        if not math.isfinite(self.std_rad) or self.std_rad < 0:
            raise ValueError("Noise standard deviation must be finite and nonnegative")
        self.seed = int(seed) + 200000
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def apply(self, action, std_rad=None):
        scale = self.std_rad if std_rad is None else float(std_rad)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Noise standard deviation must be finite and nonnegative')
        if scale == 0:
            return action
        noise = torch.randn(action.shape, generator=self.generator, device="cpu",
                            dtype=action.dtype).to(action.device)
        return action + noise * scale

    def state_dict(self):
        return dict(std_rad=self.std_rad, seed=self.seed, rng_state=self.generator.get_state())

    def load_state_dict(self, state):
        if state is None:
            if self.std_rad != 0:
                raise ValueError("Resume model has no exploration state; start a new experiment")
            return
        if state["std_rad"] != self.std_rad or state["seed"] != self.seed:
            raise ValueError("Resume exploration settings differ; start a new experiment")
        self.generator.set_state(state["rng_state"].cpu())
