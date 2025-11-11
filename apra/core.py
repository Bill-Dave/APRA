# apra/core.py
"""
APRA core adapter

This file provides a lightweight, backwards-compatible APRAEngine wrapper
that exposes the simple API used by the orchestrator and higher-level code,
while delegating to the APRA-native implementation in apra.apra_algorithm.APRAAlgorithm.

Place this file as `apra/core.py` inside the same package that contains
`apra/apra_algorithm.py` and `apra/utils.py`.

Provided API (compatibility):
 - APRAEngine(outcomes, priors, order_models, order_interaction=None, seed=None)
 - compute_posterior_from_prefix(observed_prefix, refinement_iters=4, verbose=False) -> np.ndarray
 - sample_true_outcome() -> int
 - sample_observed_features_for_outcome(outcome_idx, order_idxs=None) -> List[int]
 - get_priors() -> np.ndarray
 - set_priors(priors)
 - set_order_model(order_idx, probs)
 - add_order_model(order_idx, probs)
 - set_order_interaction(W)
 - to_json() / from_json()
 - copy()

The wrapper aims to be a drop-in replacement for the simple APRAEngine used
earlier while giving you the full APRAAlgorithm power.
"""

from __future__ import annotations
import json
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Try relative import of the APRAAlgorithm implementation
try:
    from .apra_algorithm import APRAAlgorithm
except Exception:
    # Fallback for different import contexts
    from apra.apra_algorithm import APRAAlgorithm  # type: ignore

# -------------------------
# Adapter / compatibility wrapper
# -------------------------
class APRAEngine:
    """
    Compatibility wrapper around APRAAlgorithm.

    Parameters
    ----------
    outcomes : List[str]
        Names of possible outcomes.
    priors : List[float]
        Prior probabilities for outcomes (will be normalized).
    order_models : Optional[Dict[int, List[float]]]
        Maps order index -> P(feature=1 | outcome) arrays (length K).
    order_interaction : Optional[np.ndarray]
        Square interaction matrix W (M x M) where M is number of orders.
    seed : Optional[int]
        RNG seed for deterministic sampling.
    """

    def __init__(self,
                 outcomes: List[str],
                 priors: List[float],
                 order_models: Optional[Dict[int, List[float]]] = None,
                 order_interaction: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        # instantiate underlying APRAAlgorithm
        self._apra = APRAAlgorithm(outcomes=outcomes,
                                   priors=priors,
                                   order_models=order_models,
                                   order_interaction=order_interaction,
                                   seed=seed)
        # expose convenient attributes
        self.outcomes: List[str] = list(outcomes)
        self.K: int = len(self.outcomes)

    # ---------- inference ----------
    def compute_posterior_from_prefix(self, observed_prefix: List[Tuple[int, int]], refinement_iters: int = 4, verbose: bool = False) -> np.ndarray:
        """
        Compute posterior over outcomes given observed prefix of (order_idx, bit).
        Returns a numpy probability vector (length K).
        """
        posterior = self._apra.compute_posterior_from_prefix(observed_prefix, refinement_iters=refinement_iters, verbose=verbose)
        # ensure shape and numeric type
        arr = np.array(posterior, dtype=float)
        # normalize defensively
        s = float(arr.sum())
        if s <= 0:
            arr = np.ones_like(arr) / arr.shape[0]
        else:
            arr = arr / (s + 1e-12)
        return arr

    # ---------- sampling ----------
    def sample_true_outcome(self) -> int:
        """Sample a latent true outcome index according to current prior."""
        return int(self._apra.sample_true_outcome())

    def sample_observed_features_for_outcome(self, outcome_idx: int, order_idxs: Optional[List[int]] = None) -> List[int]:
        """
        For the given outcome index, sample observed feature bits.
        If order_idxs is provided, the returned list corresponds to that order list.
        Otherwise returns bits for all known orders in APRA's order list.
        """
        sampled = self._apra.sample_observed_features_for_outcome(outcome_idx, order_idxs=order_idxs)
        # APRAAlgorithm returns dict(order->bit) when given order_idxs is None, or dict(order->bit)
        if isinstance(sampled, dict):
            if order_idxs is None:
                # return in canonical order order_index_list if available
                try:
                    order_list = self._apra.order_index_list
                except Exception:
                    order_list = sorted(list(sampled.keys()))
                return [int(sampled[o]) for o in order_list]
            else:
                return [int(sampled[o]) for o in order_idxs]
        # If it's already a list, return as list
        if isinstance(sampled, list) or isinstance(sampled, tuple):
            return list(sampled)
        # best-effort fallback: convert to list of values
        try:
            return list(sampled)
        except Exception:
            raise RuntimeError("Unexpected sampled features format from APRAAlgorithm")

    # ---------- priors & order model management ----------
    def get_priors(self) -> np.ndarray:
        """Return current priors as numpy array (normalized)."""
        # APRAAlgorithm stores log_prior internally; we can emulate via computing posterior with no evidence
        # but APRAAlgorithm doesn't expose log_prior directly - rely on computing posterior with empty observed prefix
        p = self.compute_posterior_from_prefix([])
        return np.array(p, dtype=float)

    def set_priors(self, priors: List[float], normalize: bool = True) -> None:
        """Set new priors (list of length K)."""
        # APRAAlgorithm does not expose a setter method; rebuild internal APRAAlgorithm with new priors while preserving orders and W
        state = {
            "outcomes": self._apra.outcomes,
            "log_prior": np.log(np.array(priors, dtype=float) / (float(np.sum(priors)) + 1e-12)).tolist(),
            "order_models": {int(k): v.tolist() for k, v in self._apra.order_models.items()},
            "W": getattr(self._apra, "W", None).tolist() if getattr(self._apra, "W", None) is not None else None
        }
        # Recreate underlying apra with new priors (APRAAlgorithm expects priors in normal prob space)
        pri_arr = list(priors)
        self._apra = APRAAlgorithm(outcomes=state["outcomes"], priors=pri_arr, order_models=state["order_models"], order_interaction=(np.array(state["W"]) if state["W"] is not None else None))

    def set_order_model(self, order_idx: int, probs: List[float]) -> None:
        """Set/replace an order model (P(feature=1|outcome))."""
        self._apra.add_order_model(order_idx, probs)

    def add_order_model(self, order_idx: int, probs: List[float]) -> None:
        """Add an order model (alias)."""
        self._apra.add_order_model(order_idx, probs)

    def set_order_interaction(self, W: Any) -> None:
        """
        Set the order interaction matrix W (square).
        Accepts numpy array-like or list-of-lists.
        """
        import numpy as _np
        Warr = _np.array(W, dtype=float)
        self._apra.set_order_interaction(Warr)

    # ---------- serialization ----------
    def to_json(self) -> str:
        """
        Serialize wrapper state to JSON. Uses the underlying APRAAlgorithm to_json method.
        """
        return self._apra.to_json()

    @staticmethod
    def from_json(s: str) -> "APRAEngine":
        """
        Recreate APRAEngine from JSON produced by APRAAlgorithm.to_json.
        """
        apra = APRAAlgorithm.from_json(s)
        engine = APRAEngine(outcomes=apra.outcomes, priors=[float(x) for x in np.exp(apra.log_prior)], order_models={int(k): v.tolist() for k, v in apra.order_models.items()}, order_interaction=getattr(apra, "W", None))
        # replace underlying algorithm directly so we keep the exact object
        engine._apra = apra
        engine.outcomes = list(apra.outcomes)
        engine.K = len(engine.outcomes)
        return engine

    def copy(self) -> "APRAEngine":
        """Deep copy of the engine."""
        new = deepcopy(self)
        return new

    # ---------- debug helpers ----------
    def pretty_prior(self) -> Dict[str, float]:
        p = self.get_priors()
        return {self.outcomes[i]: float(p[i]) for i in range(self.K)}

    def pretty_order_model(self, order_idx: int) -> Dict[str, float]:
        if order_idx not in self._apra.order_models:
            raise KeyError(f"order {order_idx} not found in order models")
        v = self._apra.order_models[int(order_idx)]
        return {self.outcomes[i]: float(v[i]) for i in range(self.K)}

# -------------------------
# Simple smoke demo when run directly
# -------------------------
if __name__ == "__main__":
    # Demo to verify compatibility
    outcomes = ["django_drf", "fastapi", "flask_minimal"]
    priors = [0.5, 0.3, 0.2]
    order_models = {
        0: [0.9, 0.6, 0.2],   # security_critical
        1: [0.4, 0.8, 0.2]    # performance_need
    }
    W = [[0.0, 0.25], [0.0, 0.0]]
    engine = APRAEngine(outcomes, priors, order_models, order_interaction=W, seed=42)
    print("Priors:", engine.pretty_prior())
    obs = [(0, 1)]
    post = engine.compute_posterior_from_prefix(obs, refinement_iters=4, verbose=True)
    print("Posterior after observing order 0=1:", post.tolist())
    s = engine.sample_observed_features_for_outcome(0, order_idxs=[0,1])
    print("Sampled features for outcome 0 on orders [0,1]:", s)
    print("Serialized JSON length:", len(engine.to_json()))
```0