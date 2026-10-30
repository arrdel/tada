# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""Trainable beta controller for trust-, confidence-, and availability-aware routing.

In PPAI and baseline TADA, the congestion weight `beta` in
    score(i, j) = eff_trust(j, k) - beta * eff_load(j)
is a fixed scalar hyperparameter. Here we make `beta` a learnable function
`beta_theta(s)` of the live network/trust/confidence/availability state `s`:

  1. Online Policy-Gradient (`TrainableBetaController`):
     Learns `theta` online via Gaussian policy gradient with an EMA reward
     baseline during discrete-event simulation, or deploys weights trained
     offline on GPU.
  2. GPU Neural Actor-Critic Trainer (`train_beta_policy_torch`):
     Trains a contextual `beta_theta(s)` network in PyTorch on CUDA over real
     LLM trace ensembles, dynamic M/M/1 queue states, and peer availability
     dropouts.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "bias",
    "mean_load",
    "load_dispersion",
    "top1_load",
    "trust_gap_top1_top3",
    "unavailability",
    "confidence_uncertainty",
    "cluster_norm",
]
N_FEATURES = len(FEATURE_NAMES)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def extract_beta_features(
    trust_s: np.ndarray,
    load_s: np.ndarray,
    avail_s: np.ndarray,
    conf_s: np.ndarray,
    k: int,
    n_clusters: int = 5,
) -> np.ndarray:
    """Compute the 8-D routing context feature vector phi(s) for shortlist peers."""
    if len(trust_s) == 0:
        return np.zeros(N_FEATURES, dtype=float)
    order = np.argsort(-trust_s)
    top_idx = order[: min(5, len(order))]
    top1 = order[0]
    top3 = order[min(2, len(order) - 1)]

    mean_load = float(np.mean(load_s[top_idx]))
    std_load = float(np.std(load_s[top_idx]))
    top1_load = float(load_s[top1])
    trust_gap = float(max(0.0, trust_s[top1] - trust_s[top3]))
    mean_avail = float(np.mean(avail_s[top_idx]))
    mean_conf = float(np.mean(conf_s[top_idx]))

    return np.array(
        [
            1.0,
            min(mean_load / 0.15, 3.0),
            min(std_load / 0.10, 3.0),
            min(top1_load / 0.15, 3.0),
            min(5.0 * trust_gap, 2.5),
            1.0 - np.clip(mean_avail, 0.0, 1.0),
            1.0 - np.clip(mean_conf, 0.0, 1.0),
            float(k) / max(n_clusters - 1, 1),
        ],
        dtype=float,
    )


class TrainableBetaController:
    """Contextual beta_theta(s) controller supporting linear or 2-layer MLP weights
    and online REINFORCE policy-gradient updates."""

    def __init__(
        self,
        beta_min: float = 0.2,
        beta_max: float = 12.0,
        init_beta: float = 4.0,
        lr: float = 0.04,
        explore_std: float = 0.25,
        train: bool = False,
        weights: dict | np.ndarray | list | None = None,
    ):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.lr = float(lr)
        self.explore_std = float(explore_std)
        self.train = bool(train)
        self.baseline = 0.65
        self.baseline_alpha = 0.05
        self.mlp = None

        # Default physics-informed warm start:
        # positive on load dispersion & top1_load, negative on trust_gap
        frac0 = np.clip((init_beta - beta_min) / max(beta_max - beta_min, 1e-6), 0.05, 0.95)
        b0 = float(np.log(frac0 / (1.0 - frac0)))
        self.w = np.array([b0, 0.35, 0.55, 0.45, -0.85, 0.30, -0.25, 0.0], dtype=float)

        if weights is not None:
            if isinstance(weights, dict) and "w1" in weights:
                self.mlp = {k: np.asarray(v, dtype=float) for k, v in weights.items()}
                if "w_linear" in weights:
                    self.w = np.asarray(weights["w_linear"], dtype=float).copy()
            else:
                arr = np.asarray(weights, dtype=float).ravel()
                if len(arr) == N_FEATURES:
                    self.w = arr.copy()

        self.history_beta: list[float] = []

    def _forward_logit(self, phi: np.ndarray) -> float:
        if self.mlp is not None and not self.train:
            h = np.tanh(phi @ self.mlp["w1"] + self.mlp["b1"])
            if "w2" in self.mlp and self.mlp["w2"].ndim == 2:
                h2 = np.tanh(h @ self.mlp["w2"] + self.mlp["b2"])
                out = float(np.ravel(h2 @ self.mlp["w3"] + self.mlp["b3"])[0])
            else:
                out = float(np.ravel(h @ self.mlp["w2"] + self.mlp["b2"])[0])
            return out
        return float(np.dot(self.w, phi))

    def select_beta(self, phi: np.ndarray, rng: np.random.Generator) -> tuple[float, dict]:
        mu_z = self._forward_logit(phi)
        if self.train and self.explore_std > 0.0:
            eps = float(rng.normal(0.0, self.explore_std))
            z = mu_z + eps
        else:
            eps = 0.0
            z = mu_z
        sig = float(_sigmoid(z))
        beta = self.beta_min + (self.beta_max - self.beta_min) * sig
        self.history_beta.append(float(beta))
        step_info = {"phi": phi, "mu_z": mu_z, "z": z, "eps": eps, "beta": float(beta)}
        return float(beta), step_info

    def update(self, step_info: dict | None, reward: float):
        if not self.train or step_info is None:
            return
        adv = float(reward) - self.baseline
        self.baseline = (1.0 - self.baseline_alpha) * self.baseline + self.baseline_alpha * float(reward)
        # Score function gradient d/dw log N(z; mu_z, sigma^2) = ((z - mu_z) / sigma^2) * phi
        sigma2 = max(self.explore_std ** 2, 1e-4)
        grad_logp = ((step_info["z"] - step_info["mu_z"]) / sigma2) * step_info["phi"]
        self.w = np.clip(self.w + self.lr * adv * grad_logp, -6.0, 6.0)

    def export_weights(self) -> dict:
        out = {"w_linear": self.w.tolist(), "baseline": float(self.baseline)}
        if self.mlp is not None:
            for k, v in self.mlp.items():
                out[k] = v.tolist()
        return out
