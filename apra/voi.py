# apra/voi.py
"""
VOI helpers for APRA kernel

This module provides Value-of-Information (VOI) utilities that work with either
the APRAAlgorithm implementation or the APRAEngine compatibility wrapper.

Key functions
-------------
- expected_utility(posterior, utilities)
    Compute expected utility given posterior over outcomes and utilities matrix
    (actions x outcomes).

- compute_voi_feature(apra, posterior, order_idx, utilities, world_model=None, mc_samples=256)
    Estimate the VOI of observing a single binary feature (order_idx).
    Uses analytical posterior update when APRA exposes the order model for the
    order; otherwise falls back to Monte-Carlo using the provided world_model.

- compute_action_value_with_rollouts(world_model, apra, posterior, action_idx, utilities,
                                     horizon=3, rollouts=200)
    Monte-Carlo estimate of expected cumulative utility when starting from a sampled
    outcome drawn from the posterior and executing an action sequence chosen by
    simple rollout policy (random or provided).

- choose_best_action(posterior, utilities)
    Returns best action index and its expected utility under current posterior.

Notes
-----
- utilities is a numpy array of shape (A, K): rows are actions, columns are outcomes.
- apra may be APRAAlgorithm (has order_models as dict) or wrapper exposing order_models.
- For high-assurance use, run more mc_samples/rollouts.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

EPS = 1e-12

# local imports guarded to avoid circulars; functions accept generic apra/world_model objects
try:
    from .world import WorldModel
except Exception:
    WorldModel = None  # type: ignore

# -------------------------
# Utility: expected utility
# -------------------------
def expected_utility(posterior: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    """
    Compute expected utility per action under given posterior.

    Args:
        posterior: array shape (K,) probabilities over outcomes.
        utilities: array shape (A, K) utilities.

    Returns:
        eu: array shape (A,) expected utility per action.
    """
    posterior = np.asarray(posterior, dtype=float)
    utilities = np.asarray(utilities, dtype=float)
    if posterior.ndim != 1:
        raise ValueError("posterior must be 1-D")
    if utilities.ndim != 2:
        raise ValueError("utilities must be 2-D (A, K)")
    if utilities.shape[1] != posterior.shape[0]:
        raise ValueError("utilities columns must match posterior length")
    return utilities.dot(posterior)

def choose_best_action(posterior: np.ndarray, utilities: np.ndarray) -> Tuple[int, float]:
    eus = expected_utility(posterior, utilities)
    best_idx = int(np.argmax(eus))
    return best_idx, float(eus[best_idx])

# -------------------------
# VOI for a single binary feature (order)
# -------------------------
def compute_voi_feature(apra: Any,
                        posterior: np.ndarray,
                        order_idx: int,
                        utilities: np.ndarray,
                        world_model: Optional[Any] = None,
                        mc_samples: int = 256) -> Tuple[float, Dict[str, float]]:
    """
    Compute Value of Information (VOI) for observing a binary feature at order_idx.

    Strategy:
      - If apra has order_models and includes order_idx:
          * compute P(f=1 | posterior) = sum_outcome P(f=1|o) * P(o)
          * compute posterior conditioned on f=1 and f=0 via Bayes rule (analytical)
          * compute best expected utility under each posterior
          * VOI = expected_best_after_observation - best_now
      - Else if world_model provided:
          * Monte Carlo sample outcomes from posterior, then sample feature given outcome via world_model
          * For each sampled feature value, compute posterior (approx via Bayes using apra order_models if present)
          * Average best_action utilities
      - Returns VOI and a diagnostic dict.

    Args:
        apra: APRAAlgorithm or wrapper exposing order_models (dict order->vector).
        posterior: (K,) posterior over outcomes.
        order_idx: integer index of order/feature to observe.
        utilities: (A, K) utility matrix.
        world_model: optional WorldModel to sample realistic features (fallback).
        mc_samples: Monte Carlo samples when needed.

    Returns:
        (voi, diag)
    """
    posterior = np.asarray(posterior, dtype=float)
    utilities = np.asarray(utilities, dtype=float)
    K = posterior.shape[0]
    best_now = float(np.max(expected_utility(posterior, utilities)))

    # try to access order model analytically
    order_models = None
    if hasattr(apra, "order_models"):
        order_models = getattr(apra, "order_models")
    elif hasattr(apra, "_apra") and hasattr(apra._apra, "order_models"):
        order_models = getattr(apra._apra, "order_models")

    if order_models is not None and int(order_idx) in order_models:
        pvec = np.asarray(order_models[int(order_idx)], dtype=float)
        # p(f=1) under current belief
        p_f1 = float(np.sum(pvec * posterior))
        p_f0 = 1.0 - p_f1
        # posterior given f=1
        log_pf1 = np.log(pvec + EPS)
        log_scores1 = np.log(posterior + EPS) + log_pf1
        post1 = np.exp(log_scores1 - np.max(log_scores1))
        post1 = post1 / (post1.sum() + EPS)
        # posterior given f=0
        log_pf0 = np.log(1.0 - pvec + EPS)
        log_scores0 = np.log(posterior + EPS) + log_pf0
        post0 = np.exp(log_scores0 - np.max(log_scores0))
        post0 = post0 / (post0.sum() + EPS)
        best1 = float(np.max(expected_utility(post1, utilities)))
        best0 = float(np.max(expected_utility(post0, utilities)))
        expected_after = p_f1 * best1 + p_f0 * best0
        voi = expected_after - best_now
        diag = {"p_f1": p_f1, "best_now": best_now, "best1": best1, "best0": best0}
        return float(voi), diag

    # fallback MC using world_model if available, else sample using a default 0.5 model
    if world_model is not None:
        sum_best = 0.0
        count = 0
        for s in range(mc_samples):
            # sample a latent outcome from posterior
            outcome = int(np.random.choice(K, p=posterior))
            # sample feature using world_model's order model for that outcome (best-effort)
            # world_model should expose apra/order_models or sample logic
            try:
                # attempt to use world_model.simulate_step as sampling oracle
                _, feats = world_model.simulate_step(outcome, action_idx=None)
                f = feats.get(order_idx, 1 if np.random.rand() < 0.5 else 0)
            except Exception:
                f = 1 if np.random.rand() < 0.5 else 0
            # compute posterior after observing f (approx using order_models if present)
            if order_models is not None and int(order_idx) in order_models:
                pvec = np.asarray(order_models[int(order_idx)], dtype=float)
                if f == 1:
                    log_pf = np.log(pvec + EPS)
                else:
                    log_pf = np.log(1.0 - pvec + EPS)
                log_scores = np.log(posterior + EPS) + log_pf
                post_v = np.exp(log_scores - np.max(log_scores))
                post_v = post_v / (post_v.sum() + EPS)
            else:
                # fallback: assume posterior unchanged (no info)
                post_v = posterior.copy()
            best_after = float(np.max(expected_utility(post_v, utilities)))
            sum_best += best_after
            count += 1
        expected_after = sum_best / max(1, count)
        voi = expected_after - best_now
        return float(voi), {"mc_samples": mc_samples, "best_now": best_now, "expected_after": expected_after}
    else:
        # no world model, no order model -> no VOI (or assume 0.5 chance)
        # simple heuristic: assume binary feature with p=0.5 and no discrimination -> VOI=0
        return 0.0, {"reason": "no_order_model_no_world_model", "best_now": best_now}

# -------------------------
# Action value estimation via rollouts
# -------------------------
def compute_action_value_with_rollouts(world_model: Any,
                                       apra: Any,
                                       posterior: np.ndarray,
                                       action_idx: int,
                                       utilities: np.ndarray,
                                       horizon: int = 3,
                                       rollouts: int = 200) -> float:
    """
    Estimate expected cumulative utility if we start by selecting action_idx and then
    follow a default random policy for remaining steps.

    Procedure:
      - For each rollout:
         * sample a latent starting outcome from posterior
         * construct an action sequence (first element = action_idx, subsequent random picks)
         * simulate rollout with world_model.simulate_rollout
         * accumulate utilities for each step using utilities[action_t][outcome_t]
      - Return average cumulative reward.

    Args:
        world_model: WorldModel instance with simulate_rollout(start, actions, horizon)
        apra: APRA object (unused here except for compatibility)
        posterior: (K,) distribution over latent starting outcomes
        action_idx: chosen initial action index
        utilities: (A, K) utilities matrix
        horizon: rollout horizon
        rollouts: number of rollouts to average

    Returns:
        average cumulative utility (float)
    """
    posterior = np.asarray(posterior, dtype=float)
    utilities = np.asarray(utilities, dtype=float)
    K = posterior.shape[0]
    A = utilities.shape[0]
    total = 0.0
    for r in range(rollouts):
        start = int(np.random.choice(K, p=posterior))
        actions = [action_idx] + [int(np.random.randint(0, A)) for _ in range(max(0, horizon - 1))]
        traj = world_model.simulate_rollout(start, actions, horizon)
        cum = 0.0
        for t, (outcome_t, feats) in enumerate(traj):
            act = actions[t]
            # safety: if utilities does not have this action (index out of range), clamp
            act_idx = int(act) if 0 <= int(act) < A else 0
            cum += float(utilities[act_idx, int(outcome_t)])
        total += cum
    return float(total / max(1, rollouts))

def choose_best_action_by_rollouts(world_model: Any,
                                   apra: Any,
                                   posterior: np.ndarray,
                                   utilities: np.ndarray,
                                   horizon: int = 3,
                                   rollouts: int = 200) -> Tuple[int, float]:
    """
    Evaluate all actions via rollouts and return the best action index and estimated value.
    """
    A = int(utilities.shape[0])
    best_val = -1e18
    best_action = 0
    for a in range(A):
        val = compute_action_value_with_rollouts(world_model, apra, posterior, a, utilities, horizon=horizon, rollouts=rollouts)
        if val > best_val:
            best_val = val
            best_action = a
    return best_action, float(best_val)

# -------------------------
# Small demo when run directly
# -------------------------
if __name__ == "__main__":
    # quick smoke test for analytical branch
    from .apra_algorithm import APRAAlgorithm
    outcomes = ["win", "lose", "draw"]
    priors = [0.5, 0.25, 0.25]
    order_models = {0: [0.8, 0.2, 0.5], 1: [0.4, 0.6, 0.3]}
    apra = APRAAlgorithm(outcomes, priors, order_models, seed=123)
    post = apra.compute_posterior_from_prefix([(0, 1)])
    utilities = np.array([[10, -100, 5], [2, 1, 3], [5, -10, 2]], dtype=float)
    voi, diag = compute_voi_feature(apra, post, 1, utilities)
    print("VOI for order 1:", voi, "diag:", diag)
```0