from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_logits = np.exp(np.clip(shifted, -60.0, 60.0))
    total = float(np.sum(exp_logits))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(logits, 1.0 / max(1, logits.size), dtype=float)
    return exp_logits / total


def _sanitize_vector(values: Sequence[float], fallback_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return np.zeros(fallback_size, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


@dataclass
class RLMutationAgentConfig:
    state_size: int
    hidden_size: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    update_epochs: int = 4
    gradient_clip_norm: float = 1.0
    mix_temperature: float = 1.0
    min_mix_weight: float = 1e-6
    seed: Optional[int] = None


class RolloutBuffer:
    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.logprobs: List[float] = []
        self.dones: List[bool] = []

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.logprobs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

    def mark_last_done(self) -> None:
        if self.dones:
            self.dones[-1] = True


class RLMutationAgent:
    def __init__(self, config: RLMutationAgentConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        scale_input = 1.0 / math.sqrt(max(1, config.state_size))
        scale_hidden = 1.0 / math.sqrt(max(1, config.hidden_size))
        self.w1 = self.rng.normal(0.0, scale_input, size=(config.state_size, config.hidden_size))
        self.b1 = np.zeros(config.hidden_size, dtype=float)
        self.w_policy = self.rng.normal(0.0, scale_hidden, size=(config.hidden_size, 3))
        self.b_policy = np.zeros(3, dtype=float)
        self.w_value = self.rng.normal(0.0, scale_hidden, size=(config.hidden_size, 1))
        self.b_value = np.zeros(1, dtype=float)

        self.buffer = RolloutBuffer()
        self.last_mix = np.array([1.0 / 3.0] * 3, dtype=float)

    def reset_episode(self) -> None:
        self.buffer.clear()

    def get_last_mix(self) -> np.ndarray:
        return self.last_mix.copy()

    def _forward(self, state: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        state_vec = _sanitize_vector(state, self.config.state_size)
        if state_vec.size != self.config.state_size:
            if state_vec.size < self.config.state_size:
                padded = np.zeros(self.config.state_size, dtype=float)
                padded[:state_vec.size] = state_vec
                state_vec = padded
            else:
                state_vec = state_vec[: self.config.state_size]
        hidden_linear = state_vec @ self.w1 + self.b1
        hidden = np.tanh(np.clip(hidden_linear, -20.0, 20.0))
        logits = (hidden @ self.w_policy) + self.b_policy
        logits = logits / max(self.config.mix_temperature, 1e-6)
        mix = _stable_softmax(logits)
        value = float(np.squeeze((hidden @ self.w_value) + self.b_value))
        return state_vec, hidden, value, mix

    def _mix_logprob(self, action: np.ndarray, reference_mix: np.ndarray) -> float:
        safe_action = np.clip(action, self.config.min_mix_weight, 1.0)
        safe_action = safe_action / max(np.sum(safe_action), self.config.min_mix_weight)
        safe_reference = np.clip(reference_mix, self.config.min_mix_weight, 1.0)
        safe_reference = safe_reference / max(np.sum(safe_reference), self.config.min_mix_weight)
        return float(np.sum(safe_action * np.log(safe_reference)))

    def act(self, state: Sequence[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        _, _, value, mix = self._forward(state)
        mix = np.clip(mix, self.config.min_mix_weight, 1.0)
        mix = mix / max(np.sum(mix), self.config.min_mix_weight)
        self.last_mix = mix
        return mix.copy(), {"value": value, "logprob": self._mix_logprob(mix, mix)}

    def store_transition(
        self,
        state: Sequence[float],
        action: Sequence[float],
        reward: float,
        value: float,
        logprob: float,
        done: bool,
    ) -> None:
        self.buffer.states.append(_sanitize_vector(state, self.config.state_size))
        action_vec = np.clip(_sanitize_vector(action, 3), self.config.min_mix_weight, 1.0)
        action_sum = float(np.sum(action_vec))
        if action_sum <= 0.0 or not np.isfinite(action_sum):
            action_vec = np.array([1.0 / 3.0] * 3, dtype=float)
        else:
            action_vec = action_vec / action_sum
        self.buffer.actions.append(action_vec)
        self.buffer.rewards.append(float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)))
        self.buffer.values.append(float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)))
        self.buffer.logprobs.append(float(np.nan_to_num(logprob, nan=0.0, posinf=0.0, neginf=0.0)))
        self.buffer.dones.append(bool(done))

    def end_episode(self) -> None:
        self.buffer.mark_last_done()

    def has_pending_transitions(self) -> bool:
        return len(self.buffer) > 0

    def _compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        size = len(self.buffer)
        rewards = np.asarray(self.buffer.rewards, dtype=float)
        values = np.asarray(self.buffer.values, dtype=float)
        dones = np.asarray(self.buffer.dones, dtype=bool)
        advantages = np.zeros(size, dtype=float)
        returns = np.zeros(size, dtype=float)

        next_value = 0.0
        next_advantage = 0.0
        for index in range(size - 1, -1, -1):
            mask = 0.0 if dones[index] else 1.0
            delta = rewards[index] + self.config.gamma * next_value * mask - values[index]
            next_advantage = delta + self.config.gamma * self.config.gae_lambda * mask * next_advantage
            advantages[index] = next_advantage
            returns[index] = advantages[index] + values[index]
            next_value = values[index]

        adv_std = float(np.std(advantages))
        if adv_std > 1e-8:
            advantages = (advantages - float(np.mean(advantages))) / adv_std
        else:
            advantages = advantages - float(np.mean(advantages))
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return advantages, returns

    def update(self) -> Dict[str, float]:
        if not self.has_pending_transitions():
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        advantages, returns = self._compute_advantages()
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0

        for _ in range(max(1, self.config.update_epochs)):
            grad_w1 = np.zeros_like(self.w1)
            grad_b1 = np.zeros_like(self.b1)
            grad_w_policy = np.zeros_like(self.w_policy)
            grad_b_policy = np.zeros_like(self.b_policy)
            grad_w_value = np.zeros_like(self.w_value)
            grad_b_value = np.zeros_like(self.b_value)

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0

            for index, state in enumerate(self.buffer.states):
                state_vec, hidden, value_pred, mix_pred = self._forward(state)
                action = self.buffer.actions[index]
                advantage = float(advantages[index])
                return_target = float(returns[index])
                old_logprob = float(self.buffer.logprobs[index])
                new_logprob = self._mix_logprob(action, mix_pred)
                ratio = math.exp(float(np.clip(new_logprob - old_logprob, -20.0, 20.0)))

                unclipped = ratio * advantage
                clipped_ratio = float(np.clip(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon))
                clipped = clipped_ratio * advantage
                use_clipped = (advantage >= 0.0 and unclipped > clipped) or (advantage < 0.0 and unclipped < clipped)

                policy_grad_scale = 0.0 if use_clipped else -advantage * ratio
                target_mix = np.clip(action, self.config.min_mix_weight, 1.0)
                target_mix = target_mix / max(float(np.sum(target_mix)), self.config.min_mix_weight)
                grad_logits = policy_grad_scale * (mix_pred - target_mix)
                grad_logits += self.config.entropy_coef * mix_pred * (
                    np.log(np.clip(mix_pred, self.config.min_mix_weight, 1.0))
                    - np.sum(
                        mix_pred * np.log(np.clip(mix_pred, self.config.min_mix_weight, 1.0))
                    )
                )

                value_error = value_pred - return_target
                grad_value_head = self.config.value_coef * value_error

                grad_hidden = (self.w_policy @ grad_logits) + (self.w_value[:, 0] * grad_value_head)
                grad_hidden_linear = grad_hidden * (1.0 - hidden ** 2)

                grad_w_policy += np.outer(hidden, grad_logits)
                grad_b_policy += grad_logits
                grad_w_value += np.outer(hidden, np.array([grad_value_head]))
                grad_b_value += np.array([grad_value_head])
                grad_w1 += np.outer(state_vec, grad_hidden_linear)
                grad_b1 += grad_hidden_linear

                epoch_policy_loss += -min(unclipped, clipped)
                epoch_value_loss += 0.5 * (value_error ** 2)
                epoch_entropy += -float(np.sum(mix_pred * np.log(np.clip(mix_pred, self.config.min_mix_weight, 1.0))))

            batch_size = max(1, len(self.buffer))
            gradients = [grad_w1, grad_b1, grad_w_policy, grad_b_policy, grad_w_value, grad_b_value]
            total_norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
            if total_norm > self.config.gradient_clip_norm > 0.0:
                scale = self.config.gradient_clip_norm / max(total_norm, 1e-8)
                gradients = [g * scale for g in gradients]

            step = self.config.learning_rate / batch_size
            self.w1 -= step * gradients[0]
            self.b1 -= step * gradients[1]
            self.w_policy -= step * gradients[2]
            self.b_policy -= step * gradients[3]
            self.w_value -= step * gradients[4]
            self.b_value -= step * gradients[5]

            policy_loss_total += epoch_policy_loss / batch_size
            value_loss_total += epoch_value_loss / batch_size
            entropy_total += epoch_entropy / batch_size

        metrics = {
            "policy_loss": policy_loss_total / max(1, self.config.update_epochs),
            "value_loss": value_loss_total / max(1, self.config.update_epochs),
            "entropy": entropy_total / max(1, self.config.update_epochs),
        }
        self.buffer.clear()
        return metrics
