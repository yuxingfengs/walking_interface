"""Causal command-segment velocity history shared by online and offline scoring.

Only completed physical steps enter the window. Zero observations are not padded
into a partial window. The mean is privileged Critic context, not Actor State.
Three means reduce hidden history but do not encode the complete 40-step queue.
"""
from collections import deque
import numpy as np
from .core import OBS_DIM, vector


class CommandVelocityWindow:
    def __init__(self, samples=40):
        if type(samples) is not int or samples < 1:
            raise ValueError('Reward history requires a positive integer window')
        self.samples = samples
        self.reset()

    def reset(self):
        self.command = None
        self.history = deque(maxlen=self.samples)

    def prepare(self, command):
        command = vector(command, 3, 'Reward history command')
        if self.command is None or not np.array_equal(command, self.command):
            self.history.clear()
            self.command = command.copy()
        return self.value

    @property
    def value(self):
        if not self.history:
            return np.zeros(3, np.float32)
        return np.mean(np.asarray(self.history, dtype=np.float64), axis=0).astype(np.float32)

    @property
    def count(self):
        return len(self.history)

    def advance(self, velocity, command):
        self.prepare(command)
        self.history.append(vector(velocity, 3, 'Reward history velocity').copy())
        return self.value


def sequence_context(arrays, cfg):
    """Reconstruct source/next means and post-action Reward inputs in order.

    Reject partial/discontinuous starts whose missing prefix cannot be recovered.
    Command changes clear history before the next State's features are formed;
    the completed transition is always scored against its source command.
    """
    obs = np.asarray(arrays.get('observations', arrays.get('obs')))
    nxt = np.asarray(arrays.get('next_observations', arrays.get('next_obs')))
    if obs.ndim != 2 or obs.shape[1:] != (OBS_DIM,) or nxt.shape != obs.shape or not len(obs):
        raise ValueError('Causal Reward requires aligned nonempty State50 sequences')
    if not np.isfinite(obs).all() or not np.isfinite(nxt).all():
        raise ValueError('Nonfinite causal Reward sequence')
    terminal = np.asarray(arrays.get('terminals', arrays.get('terminal', np.zeros(len(obs))))).reshape(-1)
    if terminal.shape != (len(obs),) or not np.isin(terminal, (0, 1)).all():
        raise ValueError('Invalid sequence terminal mask')
    window = CommandVelocityWindow(cfg['velocity_tracking']['window_samples'])
    source, following, velocities = [np.empty((len(obs), 3), np.float32) for _ in range(3)]
    counts = np.empty(len(obs), np.int32)
    scale = cfg['environment']['yaw_rate_scale']
    for i in range(len(obs)):
        reset = i == 0 or bool(terminal[i-1]) or not np.allclose(obs[i], nxt[i-1], rtol=0, atol=1e-6)
        if reset:
            if (not np.allclose(obs[i,34:46], 0., atol=1e-6)
                    or abs(float(obs[i,9])) > 1e-6 or abs(float(obs[i,8])-1) > 1e-6):
                raise ValueError('Cannot reconstruct Reward history from a partial/discontinuous trajectory')
            window.reset()
        source[i] = window.prepare(obs[i,5:8])
        velocities[i] = window.advance([nxt[i,47], nxt[i,48], nxt[i,46]/scale], obs[i,5:8])
        counts[i] = window.count
        following[i] = window.prepare(nxt[i,5:8])
    return dict(critic_features=source, next_critic_features=following,
                reward_velocities=velocities, reward_history_count=counts)
