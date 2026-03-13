"""
agent.py — self-contained Agent for inference.
No dependency on lost_cities_env_fast2 or the C extension.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class BatchedAgent(ABC):
    @abstractmethod
    def act_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray: pass


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None,
                 masks: Optional[torch.BoolTensor] = None):
        self.masks = masks
        if masks is None:
            super().__init__(probs, logits, validate_args)
        else:
            _masks = masks.to(probs.device if probs is not None else logits.device)
            self.masks = _masks
            assert logits is not None
            logits = torch.where(_masks, logits, torch.tensor(-1e8))
            super().__init__(probs, logits, validate_args)

    def entropy(self):
        if self.masks is None:
            return super().entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.zeros_like(p_log_p))
        return -p_log_p.sum(-1)


class Agent(BatchedAgent, nn.Module):
    # Fixed dimensions for the Lost Cities environment
    OBS_SIZE = 301
    NUM_ACTIONS = 600

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.OBS_SIZE, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.OBS_SIZE, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.NUM_ACTIONS), std=0.01),
        )

    @torch.no_grad()
    def act_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        x = torch.tensor(obs_batch, dtype=torch.float).to(self.device)
        mask = torch.tensor(mask_batch, dtype=torch.bool).to(self.device)
        probs = CategoricalMasked(logits=self.actor(x), masks=mask)
        return probs.sample().cpu().numpy()

    @torch.no_grad()
    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        return int(self.act_batch(obs[np.newaxis], action_mask[np.newaxis])[0])
