"""Independent continuous-action SAC implementation for the Turn task."""
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .core import CONTRACT, critic_extra_dim, project_action
from .network_inputs import (
    ProjectedCritic,
    actor_input,
    actor_input_dim,
    critic_input_dim,
    observation_dim,
)


def _mlp(input_dim, hidden, output_dim):
    layers = []
    width = input_dim
    for next_width in hidden:
        layers.extend((nn.Linear(width, next_width), nn.ReLU()))
        width = next_width
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Tanh-Gaussian policy over the 11 independent NAO joint actions."""

    independent_action_dim = 11

    def __init__(self, config, spec):
        super().__init__()
        self.config = config
        self.backbone = _mlp(actor_input_dim(config), spec['hidden'], spec['hidden'][-1])
        self.mean = nn.Linear(spec['hidden'][-1], self.independent_action_dim)
        self.log_std = nn.Linear(spec['hidden'][-1], self.independent_action_dim)
        self.log_std_min = float(spec['log_std_min'])
        self.log_std_max = float(spec['log_std_max'])

    @staticmethod
    def expand_action(independent):
        action = torch.empty(
            (*independent.shape[:-1], 12), dtype=independent.dtype,
            device=independent.device,
        )
        action[..., 0] = independent[..., 0]
        action[..., 6] = independent[..., 0]
        action[..., 1:6] = independent[..., 1:6]
        action[..., 7:12] = independent[..., 6:11]
        return action

    def distribution(self, physical_obs):
        state = actor_input(physical_obs, self.config)
        features = self.backbone(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, physical_obs, deterministic=False):
        mean, log_std = self.distribution(physical_obs)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else distribution.rsample()
        independent = torch.tanh(pre_tanh)
        action = self.expand_action(independent)
        if deterministic:
            return action, None, std
        log_prob = distribution.log_prob(pre_tanh)
        log_prob -= torch.log(1.0 - independent.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), std

    def forward(self, physical_obs):
        return self.sample(physical_obs, deterministic=True)[0]


class TwinQ(nn.Module):
    """Twin critics with the project's packed physical State/Action interface."""

    def __init__(self, config, spec):
        super().__init__()
        extra = critic_extra_dim(config)
        input_dim = critic_input_dim(config, extra)
        self.q1 = ProjectedCritic(
            list(_mlp(input_dim, spec['hidden'], 1).children()),
            extra, observation_dim(config), _actor_indices(config))
        self.q2 = ProjectedCritic(
            list(_mlp(input_dim, spec['hidden'], 1).children()),
            extra, observation_dim(config), _actor_indices(config))

    def values(self, physical_obs, action, features=None):
        parts = [physical_obs]
        if features is not None:
            parts.append(features)
        parts.append(action)
        packed = torch.cat(parts, dim=-1)
        return self.q1(packed), self.q2(packed)

    def forward(self, packed):
        return self.q1(packed)


def _actor_indices(config):
    from .network_inputs import actor_indices, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY, SHARED_VELOCITY_EMA
    if config.get('actor_observation') in (
            SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY):
        return actor_indices(config)
    return None


class SACReplay:
    """FIFO Replay owned by SAC; accepts only online transitions."""

    def __init__(self, capacity, seed, state_dim, feature_dim=0):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('SAC Replay capacity must be positive')
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.cursor = 0
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.obs = np.empty((capacity, state_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty((capacity, 12), np.float32)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.terminal = np.empty((capacity, 1), np.float32)
        if feature_dim:
            self.critic_features = np.empty((capacity, feature_dim), np.float32)
            self.next_critic_features = np.empty_like(self.critic_features)
        self.last_sample_counts = {}
        self.online = self

    def add(self, obs, action, reward, next_obs, terminated, **kwargs):
        if kwargs.get('expert', False):
            raise ValueError('SAC does not accept expert or DDPG Replay transitions')
        index = self.cursor
        self.obs[index] = obs
        self.actions[index] = project_action(action)
        self.rewards[index] = float(reward)
        self.next_obs[index] = next_obs
        self.terminal[index] = float(bool(terminated))
        if self.feature_dim:
            current = kwargs.get('critic_features')
            following = kwargs.get('next_critic_features')
            if current is None or following is None:
                raise ValueError('SAC Replay requires aligned Critic features')
            self.critic_features[index] = current
            self.next_critic_features[index] = following
        elif kwargs.get('critic_features') is not None or kwargs.get('next_critic_features') is not None:
            raise ValueError('Unexpected privileged Critic features for SAC Replay')
        self.cursor = (index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def ready(self, batch_size, critic_only=False):
        return self.count >= int(batch_size)

    def sample(self, batch_size, **kwargs):
        if not self.ready(batch_size):
            raise ValueError('SAC Replay is not ready')
        indices = self.rng.integers(self.count, size=batch_size)
        arrays = (
            self.obs[indices], self.actions[indices], self.rewards[indices],
            self.next_obs[indices], self.terminal[indices],
        )
        if self.feature_dim:
            arrays += (self.critic_features[indices], self.next_critic_features[indices])
        self.last_sample_counts = {'online': int(batch_size)}
        return tuple(torch.from_numpy(array.copy()) for array in arrays)


class SAC:
    """SAC agent independent of DDPG Actor, Critic, Replay, and checkpoints."""

    def __init__(self, config):
        if config.get('algorithm') != 'sac':
            raise ValueError('SAC agent requires algorithm="sac"')
        if config.get('expert_pipeline') or config.get('three_class_online'):
            raise ValueError('SAC Turn must use online-only experience')
        if config.get('initialization', {}).get('use_bc', False):
            raise ValueError('SAC Turn must initialize from its own random policy')
        self.config = config
        self.spec = config['sac']
        torch.manual_seed(int(config['seed']))
        torch.set_num_threads(1)

        self.actor = SquashedGaussianActor(config, self.spec)
        self.critic = TwinQ(config, self.spec)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(self.spec['actor_lr']))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(self.spec['critic_lr']))
        self.log_alpha = torch.tensor(
            np.log(float(self.spec['initial_alpha'])), dtype=torch.float32,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(self.spec['alpha_lr']))
        self.target_entropy = float(self.spec.get('target_entropy', -11.0))
        self.replay = SACReplay(
            config['training']['replay_size'], int(config['seed']) + 1,
            observation_dim(config), critic_extra_dim(config),
        )
        self.rng = np.random.default_rng(int(config['seed']) + 2)
        self.learning_starts = int(self.spec['learning_starts'])
        self._step = 0
        self.updates = 0
        self.warmup_completed = 0
        self.pretrained_critic_updates = 0
        self.bc_state = None
        self.noise_std = 1.0

    @property
    def alpha(self):
        return self.log_alpha.detach().exp()

    def schedule(self, step):
        self._step = int(step)

    def act(self, obs, explore=False):
        if explore and self._step <= self.learning_starts:
            independent = self.rng.uniform(-1., 1., size=(11,)).astype(np.float32)
            return SquashedGaussianActor.expand_action(
                torch.from_numpy(independent)).numpy()
        physical_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, std = self.actor.sample(physical_obs, deterministic=not explore)
        self.noise_std = float(std.mean().item())
        return action[0].cpu().numpy()

    def update(self, critic_only=False, update_actor=True):
        batch = self.replay.sample(self.config['training']['batch_size'])
        obs, actions, rewards, next_obs, terminal = batch[:5]
        features = batch[5] if self.replay.feature_dim else None
        next_features = batch[6] if self.replay.feature_dim else None
        alpha = self.alpha
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target.values(
                next_obs, next_action, next_features)
            target_value = torch.minimum(target_q1, target_q2) - alpha * next_log_prob
            target = rewards + float(self.spec['gamma']) * (1. - terminal) * target_value

        current_q1, current_q2 = self.critic.values(obs, actions, features)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('Nonfinite SAC twin-Critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 1.0, error_if_nonfinite=True)
        self.critic_optimizer.step()

        actor_loss_value = None
        entropy_loss_value = None
        policy_entropy = None
        actor_grad = None
        if update_actor and not critic_only:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            sampled_action, log_prob, _ = self.actor.sample(obs)
            q1_pi, q2_pi = self.critic.values(obs, sampled_action, features)
            actor_loss = (self.alpha * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError('Nonfinite SAC Actor loss')
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 1.0, error_if_nonfinite=True)
            self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            if not torch.isfinite(alpha_loss):
                raise FloatingPointError('Nonfinite SAC temperature loss')
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-10., 5.)
            actor_loss_value = float(actor_loss.detach())
            entropy_loss_value = float(alpha_loss.detach())
            policy_entropy = float((-log_prob.detach()).mean())

        with torch.no_grad():
            tau = float(self.spec['tau'])
            for source, target_net in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_net.lerp_(source, tau)
        self.updates += 1
        return dict(
            actor_loss=actor_loss_value,
            actor_rl_loss=actor_loss_value,
            actor_bc_loss=None,
            critic_loss=float(critic_loss.detach()),
            updates=self.updates,
            q_mean=float(torch.minimum(current_q1, current_q2).detach().mean()),
            target_q_mean=float(target.mean()),
            td_abs_mean=float((torch.minimum(current_q1, current_q2).detach() - target).abs().mean()),
            critic_grad_norm=float(critic_grad),
            actor_grad_norm=float(actor_grad) if actor_grad is not None else None,
            entropy_loss=entropy_loss_value,
            policy_entropy=policy_entropy,
            entropy_coefficient=float(self.alpha.item()),
        )

    def save(self, path, steps):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            kind='sac_turn_checkpoint_v1', contract=CONTRACT, config=self.config,
            env_steps=int(steps), updates=self.updates,
            actor=self.actor.state_dict(), critic=self.critic.state_dict(),
            critic_target=self.critic_target.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            log_alpha=self.log_alpha.detach().clone(),
            alpha_optimizer=self.alpha_optimizer.state_dict(),
            sac_spec=self.spec,
        ), path)
