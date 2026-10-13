"""Velocity estimator and RSL-RL integration for the Go2 task.

The actor receives 275 proprioceptive values followed by estimated base velocity.
The critic still receives the 570-dimensional privileged observation directly.
"""

from __future__ import annotations

import os
import copy

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class VelocityEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(275, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 3),
        )
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0.0, 0.1)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Go2EstimatorVecEnvWrapper(RslRlVecEnvWrapper):
    """Adds estimated velocity to actor observations and trains on privileged velocity labels."""

    def __init__(self, env, clip_actions=None, train_estimator: bool = False):
        self.estimator = VelocityEstimator().to(env.unwrapped.device)
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr=2.0e-3)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.2, total_iters=2000
        )
        self.train_estimator = train_estimator
        self.replay_inputs = torch.zeros(4096 * 3, 275, device=env.unwrapped.device)
        self.replay_targets = torch.zeros(4096 * 3, 3, device=env.unwrapped.device)
        self.replay_size = 0
        self.replay_index = 0
        self.rollout_step = 0
        super().__init__(env, clip_actions=clip_actions)

    def _augment(self, obs):
        with torch.no_grad():
            estimate = self.estimator(obs["policy"])
        obs["policy"] = torch.cat((obs["policy"], estimate), dim=-1)
        return obs

    def get_observations(self):
        return self._augment(super().get_observations())

    def step(self, actions):
        obs, reward, done, extras = super().step(actions)
        if self.train_estimator:
            self._collect(obs["policy"], obs["critic"][:, 55:58])
            self.rollout_step += 1
            if self.rollout_step % 24 == 0:
                loss = self._update_estimator()
                if loss is not None:
                    extras.setdefault("log", {})["Estimator/loss"] = loss
                    extras["log"]["Estimator/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return self._augment(obs), reward, done, extras

    def _collect(self, inputs: torch.Tensor, targets: torch.Tensor):
        count = len(inputs)
        if count >= len(self.replay_inputs):
            inputs = inputs[-len(self.replay_inputs):]
            targets = targets[-len(self.replay_targets):]
            count = len(inputs)
        first = min(count, len(self.replay_inputs) - self.replay_index)
        self.replay_inputs[self.replay_index:self.replay_index + first] = inputs[:first]
        self.replay_targets[self.replay_index:self.replay_index + first] = targets[:first]
        if first < count:
            self.replay_inputs[:count - first] = inputs[first:]
            self.replay_targets[:count - first] = targets[first:]
        self.replay_index = (self.replay_index + count) % len(self.replay_inputs)
        self.replay_size = min(self.replay_size + count, len(self.replay_inputs))

    def _update_estimator(self):
        if self.replay_size < 1024:
            return None
        # OnPolicyRunner collects actions under inference_mode; temporarily enable gradients.
        with torch.inference_mode(False), torch.enable_grad():
            order = torch.randperm(self.replay_size, device=self.device)
            losses = []
            for ids in order.split(1024):
                inputs = self.replay_inputs[ids].clone()
                targets = self.replay_targets[ids].clone()
                loss = torch.nn.functional.mse_loss(self.estimator(inputs), targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach())
            self.scheduler.step()
            return torch.stack(losses).mean().item()


class Go2EstimatorRunner(OnPolicyRunner):
    """Saves the estimator alongside PPO so resume and playback use the same actor input."""

    def save(self, path: str, infos: dict | None = None):
        saved = self.alg.save()
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = infos
        saved["velocity_estimator"] = self.env.estimator.state_dict()
        saved["velocity_estimator_optimizer"] = self.env.optimizer.state_dict()
        saved["velocity_estimator_scheduler"] = self.env.scheduler.state_dict()
        torch.save(saved, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        saved = torch.load(path, weights_only=False, map_location=map_location)
        if "velocity_estimator" not in saved:
            raise ValueError("Checkpoint has no velocity estimator; train this task with Go2EstimatorRunner")
        self.env.estimator.load_state_dict(saved["velocity_estimator"])
        if self.env.train_estimator:
            self.env.optimizer.load_state_dict(saved["velocity_estimator_optimizer"])
            self.env.scheduler.load_state_dict(saved["velocity_estimator_scheduler"])
        return super().load(path, load_cfg, strict, map_location)

    def export_estimator_to_jit(self, path: str):
        os.makedirs(path, exist_ok=True)
        traced = torch.jit.trace(copy.deepcopy(self.env.estimator).eval().cpu(), torch.zeros(1, 275))
        traced.save(os.path.join(path, "velocity_estimator.pt"))
