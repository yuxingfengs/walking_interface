"""Behavior-preserving extraction of the S0/S4 DDPG implementation."""

import copy
import random
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithms.algorithm_plugin import AlgorithmPlugin, AlgorithmUpdate
from checkpointing import load_torch_artifact, save_ddpg_checkpoint
from contracts import Transition, validate_contract_metadata
from models import BoundedActor


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        self.buffer = []

    def push(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("ReplayBuffer accepts Transition objects")
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_batch(self, batch_size: int):
        batch = random.sample(self.buffer, min(int(batch_size), len(self.buffer)))
        states = [item.observation.values for item in batch]
        actions = [item.action.values for item in batch]
        rewards = [item.reward for item in batch]
        next_states = [item.next_observation.values for item in batch]
        dones = [item.done for item in batch]
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = F.relu(self.bn1(self.l1(state)))
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.bn2(self.l2(x)))
        return self.l3(x)


class DDPGAlgorithm(AlgorithmPlugin):
    algorithm_id = "ddpg_v1"

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        device,
        gamma: float,
        tau: float,
        batch_size: int,
        buffer_size: int,
        task_id: str = "forward_walk_v1",
        resume_actor_lr: float = 1e-6,
        resume_critic_lr: float = 1e-6,
        critic_warmup_updates: int = 0,
        manage_critic_batchnorm: bool = False,
        sync_bn_running_stats: bool = False,
    ):
        self.manage_critic_batchnorm = bool(manage_critic_batchnorm)
        self.sync_bn_running_stats = bool(sync_bn_running_stats)
        if self.sync_bn_running_stats and not self.manage_critic_batchnorm:
            raise ValueError('BN statistics synchronization requires managed Critic modes')
        self.critic_warmup_updates = int(critic_warmup_updates)
        if self.critic_warmup_updates < 0:
            raise ValueError("critic_warmup_updates must be nonnegative")
        self.critic_warmup_completed = 0
        self.actor_update_count = 0
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.task_id = task_id
        self.state_dim = state_dim
        self.resume_actor_lr = float(resume_actor_lr)
        self.resume_critic_lr = float(resume_critic_lr)
        self.actor = self.create_actor(state_dim, action_dim).to(self.device)
        self.critic = self.create_critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = optim.Adam((p for p in self.actor.parameters() if p.requires_grad), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self._batch_size = int(batch_size)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        if self.manage_critic_batchnorm:
            self.set_evaluation_mode()

    def create_actor(self, state_dim, action_dim):
        return BoundedActor(state_dim, 256, action_dim)

    def create_critic(self, state_dim, action_dim, hidden_dim):
        return Critic(state_dim, action_dim, hidden_dim)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def experience_size(self) -> int:
        return len(self.replay_buffer)

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.push(transition)

    def clear_experience(self) -> None:
        self.replay_buffer.clear()

    def set_evaluation_mode(self) -> None:
        self.actor.eval()
        if getattr(self, 'manage_critic_batchnorm', False):
            self.critic.eval()
            self.actor_target.eval()
            self.critic_target.eval()

    @contextmanager
    def _fixed_critic(self):
        """Fixed value function for Actor gradients; do not disable autograd."""
        if not getattr(self, 'manage_critic_batchnorm', False):
            yield
            return
        modes = [(module, module.training) for module in self.critic.modules()]
        flags = [(parameter, parameter.requires_grad) for parameter in self.critic.parameters()]
        try:
            self.critic.eval()
            for parameter, _ in flags:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, enabled in flags:
                parameter.requires_grad_(enabled)
            for module, training in modes:
                module.training = training

    @torch.no_grad()
    def act(self, observation):
        state = torch.as_tensor(
            getattr(observation, "values", observation),
            dtype=torch.float32,
            device=self.device,
        )
        return self.actor(state.unsqueeze(0) if state.dim() == 1 else state).squeeze(0)

    def soft_update(self, net, net_target) -> None:
        with torch.no_grad():
            for param, target_param in zip(net.parameters(), net_target.parameters()):
                target_param.lerp_(param, self.tau)
            source_buffers = dict(net.named_buffers())
            bn_statistics = set()
            if getattr(self, 'sync_bn_running_stats', False):
                for prefix, module in net.named_modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        bn_statistics.update((prefix + '.' if prefix else '') + field
                                             for field in ('running_mean', 'running_var', 'num_batches_tracked'))
            for name, target_buffer in net_target.named_buffers():
                source_buffer = source_buffers[name]
                if name in bn_statistics:
                    target_buffer.copy_(source_buffer)
                elif target_buffer.is_floating_point():
                    target_buffer.lerp_(source_buffer, self.tau)
                else:
                    target_buffer.copy_(source_buffer)

    @staticmethod
    def _grad_global_norm(module) -> float:
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += parameter.grad.data.pow(2).sum().item()
        return total ** 0.5

    def update(self, critic_updates: int = 2) -> AlgorithmUpdate:
        if self.experience_size < self.batch_size:
            return AlgorithmUpdate()

        if getattr(self, 'manage_critic_batchnorm', False):
            self.actor_target.eval()
            self.critic_target.eval()
            self.critic.train()

        warming_up = self.critic_warmup_completed < self.critic_warmup_updates
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).view(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).view(-1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions).view(-1)
            target_q = rewards + self.gamma * (1 - dones) * next_q
        # Diagnostic forwards must not add BatchNorm updates.
        with self._fixed_critic(), torch.no_grad():
            current_q = self.critic(states, actions).view(-1)
        td_error = (target_q - current_q).detach().cpu()
        current_q_mean = float(current_q.mean().item())
        target_q_mean = float(target_q.mean().item())
        metrics = {
            "current_q_mean": current_q_mean,
            "target_q_mean": target_q_mean,
            "bellman_gap": target_q_mean - current_q_mean,
            "td_mse": float(F.mse_loss(current_q, target_q).item()),
            "td_error_mean": float(td_error.mean().item()),
            "td_error_std": float(td_error.std().item()),
        }

        critic_loss_total = 0.0
        critic_grad_norms = []
        if any(parameter.requires_grad for parameter in self.critic.parameters()):
            for _ in range(int(critic_updates)):
                with torch.no_grad():
                    next_actions = self.actor_target(next_states)
                    next_q = self.critic_target(next_states, next_actions).view(-1)
                    target_q = rewards + self.gamma * (1 - dones) * next_q
                current_q = self.critic(states, actions).view(-1)
                critic_loss = F.mse_loss(current_q, target_q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norms.append(self._grad_global_norm(self.critic))
                self.critic_optimizer.step()
                self.soft_update(self.critic, self.critic_target)
                critic_loss_total += critic_loss.item()
        average_critic_loss = critic_loss_total / int(critic_updates)

        if warming_up:
            self.critic_warmup_completed += 1
            metrics.update(training_stage="critic_warmup", actor_updated=False,
                critic_warmup_completed=self.critic_warmup_completed,
                critic_warmup_updates=self.critic_warmup_updates,
                critic_optimizer_steps=int(critic_updates), actor_optimizer_steps=0,
                actor_grad_norm=0.0,
                critic_grad_norm_mean=sum(critic_grad_norms) / max(1, len(critic_grad_norms)),
                actor_lr=float(self.actor_optimizer.param_groups[0]["lr"]),
                critic_lr=float(self.critic_optimizer.param_groups[0]["lr"]))
            return AlgorithmUpdate(actor_loss=0.0, critic_loss=float(average_critic_loss), metrics=metrics)

        metrics.update(training_stage="joint", actor_updated=True,
            critic_warmup_completed=self.critic_warmup_completed,
            critic_warmup_updates=self.critic_warmup_updates,
            critic_optimizer_steps=int(critic_updates), actor_optimizer_steps=1)
        predicted_actions = self.actor(states)
        with self._fixed_critic():
            actor_loss = -self.critic(states, predicted_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
        actor_grad_norm = self._grad_global_norm(self.actor)
        self.actor_optimizer.step()
        self.actor_update_count += 1
        self.soft_update(self.actor, self.actor_target)

        actor_action_var = float(predicted_actions.var(dim=0).mean().item())
        if not hasattr(self, "_last_actions"):
            self._last_actions = predicted_actions.detach().clone()
            actor_action_diff = None
        else:
            actor_action_diff = float(
                (predicted_actions - self._last_actions).pow(2).mean().item()
            )
            self._last_actions = predicted_actions.detach().clone()

        metrics.update({
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(average_critic_loss),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm_mean": float(
                sum(critic_grad_norms) / max(1, len(critic_grad_norms))
            ),
            "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
            "actor_action_var": actor_action_var,
            "actor_action_diff": actor_action_diff,
        })
        return AlgorithmUpdate(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(average_critic_loss),
            metrics=metrics,
        )

    def save_checkpoint(self, filepath, metadata=None):
        return save_ddpg_checkpoint(self, filepath, {
            **dict(metadata or {}),
            "critic_warmup_updates": self.critic_warmup_updates,
            "critic_warmup_completed": self.critic_warmup_completed,
            "actor_update_count": self.actor_update_count,
            "manage_critic_batchnorm": getattr(self, 'manage_critic_batchnorm', False),
            "sync_bn_running_stats": getattr(self, 'sync_bn_running_stats', False),
        })

    def load_checkpoint(self, filepath):
        checkpoint = load_torch_artifact(filepath, "cpu")
        metadata = checkpoint.get("meta", {})
        if bool(metadata.get('manage_critic_batchnorm', False)) != getattr(self, 'manage_critic_batchnorm', False):
            raise ValueError('Resume requires the saved Critic BatchNorm mode; start a new experiment from BC')
        if bool(metadata.get('sync_bn_running_stats', False)) != getattr(self, 'sync_bn_running_stats', False):
            raise ValueError('Resume requires the saved BN statistics sync setting; start a new experiment from BC')
        validate_contract_metadata(
            metadata, artifact=str(filepath), expected_task_id=self.task_id,
            expected_state_dim=self.actor.net[0].in_features
        )
        artifact_algorithm = metadata.get("algorithm_id") if isinstance(metadata, dict) else None
        if artifact_algorithm and artifact_algorithm != self.algorithm_id:
            raise ValueError(
                f"incompatible {filepath} Algorithm: {artifact_algorithm!r} "
                f"(expected {self.algorithm_id!r})"
            )
        saved_warmup = metadata.get("critic_warmup_updates")
        if saved_warmup is not None and saved_warmup != self.critic_warmup_updates:
            raise ValueError("Resume requires the saved critic_warmup_updates setting")
        # Legacy resume checkpoints have already trained: do not insert a new warmup.
        completed = int(metadata.get("critic_warmup_completed", self.critic_warmup_updates))
        if not 0 <= completed <= self.critic_warmup_updates:
            raise ValueError("Invalid saved critic warmup progress")
        self.critic_warmup_completed = completed
        self.actor_update_count = int(metadata.get("actor_update_count",
            max(0, int(metadata.get("global_step", 0)) - completed)))
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(
            checkpoint.get("actor_target_state_dict", checkpoint["actor_state_dict"])
        )
        self.critic_target.load_state_dict(
            checkpoint.get("critic_target_state_dict", checkpoint["critic_state_dict"])
        )
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        for module in (self.actor, self.critic, self.actor_target, self.critic_target):
            module.to(self.device)
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
        for parameter_group in self.actor_optimizer.param_groups:
            parameter_group["lr"] = self.resume_actor_lr
        for parameter_group in self.critic_optimizer.param_groups:
            parameter_group["lr"] = self.resume_critic_lr
        self.set_evaluation_mode()
        return metadata if isinstance(metadata, dict) else {}

    def load_actor_initialization(
        self,
        filepath,
        *,
        reset_optimizer: bool,
        actor_lr=None,
    ) -> bool:
        from pathlib import Path

        path = Path(filepath)
        if not path.is_file():
            print(f"Behavior cloning checkpoint not found: {path}")
            return False
        artifact = load_torch_artifact(path, "cpu")
        if isinstance(artifact, dict) and "actor_state_dict" in artifact:
            validate_contract_metadata(
                artifact.get("meta"), artifact=str(path), expected_task_id=self.task_id,
                expected_state_dim=self.actor.net[0].in_features
            )
            state_dict = artifact["actor_state_dict"]
        else:
            validate_contract_metadata(
                None, artifact=str(path), expected_task_id=self.task_id,
                expected_state_dim=self.actor.net[0].in_features
            )
            state_dict = artifact
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        if reset_optimizer:
            learning_rate = (
                float(actor_lr)
                if actor_lr is not None
                else float(self.actor_optimizer.param_groups[0]["lr"])
            )
            self.actor_optimizer = optim.Adam((p for p in self.actor.parameters() if p.requires_grad), lr=learning_rate)
        print(f"Loaded behavior cloning model into actor from '{path}'")
        return True
