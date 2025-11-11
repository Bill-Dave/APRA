# apra/world.py
"""
WorldModel and simulation utilities for APRA.

Responsibilities:
 - Maintain a (small) outcome transition model
 - Simulate single steps and multi-step rollouts
 - Sample observed order-features for outcomes using the APRA order models
 - Run counterfactual simulations by applying a hypothesis (mutating a copy of the APRA order models)
 - Produce simple statistics (outcome histograms, average feature incidence)

Design goals:
 - Work with either APRAAlgorithm or APRAEngine wrappers (detects order_models attribute)
 - Deterministic RNG via apra.utils.make_rng (seedable)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    # local utils (we created apra/utils.py)
    from .utils import make_rng, EPS
except Exception:
    # fallback
    import os, numpy as _np
    def make_rng(seed: Optional[int] = None) -> _np.random.RandomState:
        if seed is None:
            seed_env = os.environ.get("APRA_SEED")
            if seed_env is not None:
                try:
                    seed = int(seed_env)
                except Exception:
                    seed = None
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little") & 0xFFFFFFFF
        return _np.random.RandomState(seed)
    EPS = 1e-12

def _get_order_models(apra: Any) -> Dict[int, np.ndarray]:
    """
    Return a dict mapping order_idx -> numpy array length K giving P(feature=1 | outcome).
    Works for APRAAlgorithm, APRAEngine, or a plain object with 'order_models'.
    """
    # APRAAlgorithm stores order_models as attribute
    if hasattr(apra, "order_models"):
        om = getattr(apra, "order_models")
        # if values are lists convert to arrays
        return {int(k): (v if isinstance(v, np.ndarray) else np.array(v, dtype=float)) for k, v in om.items()}
    # fallback: maybe wrapped under _apra
    if hasattr(apra, "_apra") and hasattr(apra._apra, "order_models"):
        om = getattr(apra._apra, "order_models")
        return {int(k): (v if isinstance(v, np.ndarray) else np.array(v, dtype=float)) for k, v in om.items()}
    # no order models => empty
    return {}

def _get_K(apra: Any) -> int:
    if hasattr(apra, "K"):
        return int(apra.K)
    if hasattr(apra, "_apra") and hasattr(apra._apra, "K"):
        return int(apra._apra.K)
    # best effort: infer from an order model
    om = _get_order_models(apra)
    if len(om) > 0:
        anyvec = next(iter(om.values()))
        return int(len(anyvec))
    raise RuntimeError("Cannot determine number of outcomes K from provided 'apra' object")

class WorldModel:
    """
    Small Markov-style world model over outcomes.

    Parameters
    ----------
    apra:
        APRAAlgorithm or APRAEngine-like object providing order_models and K
    transition_matrix:
        Optional K x K numpy array. If None a default near-identity matrix is created.
    rng_seed:
        Optional RNG seed for deterministic behavior.
    action_bias_fn:
        Optional callable(action_idx:int, probs:np.ndarray) -> np.ndarray that returns modified outcome probs after action
    """
    def __init__(self,
                 apra: Any,
                 transition_matrix: Optional[np.ndarray] = None,
                 rng_seed: Optional[int] = None,
                 action_bias_fn: Optional[Any] = None):
        self.apra = apra
        self.K = _get_K(apra)
        if transition_matrix is None:
            # default: stay with high prob, small uniform off-diagonal noise
            if self.K == 1:
                self.transition = np.array([[1.0]])
            else:
                tm = np.eye(self.K) * 0.7
                off = (0.3 / (self.K - 1))
                for i in range(self.K):
                    for j in range(self.K):
                        if i != j:
                            tm[i, j] = off
                self.transition = tm
        else:
            tm = np.array(transition_matrix, dtype=float)
            if tm.shape != (self.K, self.K):
                raise ValueError("transition_matrix must be shape (K, K)")
            # normalize rows defensively
            row_sums = tm.sum(axis=1)
            tm = (tm.T / (row_sums + EPS)).T
            self.transition = tm
        self.rng = make_rng(rng_seed)
        self.action_bias_fn = action_bias_fn

    def simulate_step(self, current_outcome: int, action_idx: Optional[int] = None) -> Tuple[int, Dict[int, int]]:
        """
        Simulate one world step.
        Returns: (next_outcome_idx, observed_features_dict)
        observed_features_dict maps order_idx -> bit (0/1) sampled using APRA order_models for the next outcome.
        """
        probs = self.transition[int(current_outcome)].copy()
        if action_idx is not None:
            if self.action_bias_fn is not None:
                probs = self.action_bias_fn(action_idx, probs.copy())
            else:
                # default bias: rotate probs by action_idx mod K and slightly sharpen
                bias = int(action_idx) % max(1, self.K)
                probs = np.roll(probs, -bias)
                # slight sharpening toward top probs
                probs = probs ** 1.05
                probs = probs / (probs.sum() + EPS)
        # sample next outcome deterministically with RNG
        next_idx = int(self.rng.choice(self.K, p=probs))
        # sample features using APRA order models
        order_models = _get_order_models(self.apra)
        feats: Dict[int, int] = {}
        for oi, pvec in order_models.items():
            # ensure pvec is array length K
            if not isinstance(pvec, np.ndarray):
                pvec = np.array(pvec, dtype=float)
            p = float(pvec[next_idx])
            feats[int(oi)] = 1 if self.rng.rand() < p else 0
        return next_idx, feats

    def simulate_rollout(self, start_outcome: int, actions: List[Optional[int]], horizon: int) -> List[Tuple[int, Dict[int, int]]]:
        """
        Simulate a rollout of length 'horizon' starting from start_outcome.
        'actions' may be shorter than horizon; missing actions treated as None.
        Returns list of (outcome_idx, observed_features) for each timestep.
        """
        cur = int(start_outcome)
        traj: List[Tuple[int, Dict[int,int]]] = []
        for t in range(horizon):
            act = actions[t] if t < len(actions) else None
            cur, feats = self.simulate_step(cur, action_idx=act)
            traj.append((cur, feats))
        return traj

    def rollouts_distribution(self, start_outcome: int, policy_actions: List[int], horizon: int, num_rollouts: int = 200) -> Dict[str, Any]:
        """
        Run many rollouts using the provided action sequence (repeats if shorter).
        Returns aggregated statistics:
         - outcome_counts: histogram of final outcomes
         - avg_feature_probs: average frequency of each order being 1 across runs
        """
        final_counts = np.zeros(self.K, dtype=int)
        order_models = _get_order_models(self.apra)
        feat_sums: Dict[int, int] = {oi: 0 for oi in order_models.keys()}
        for r in range(num_rollouts):
            actions = [policy_actions[i % len(policy_actions)] for i in range(horizon)]
            traj = self.simulate_rollout(start_outcome, actions, horizon)
            final_outcome = traj[-1][0]
            final_counts[int(final_outcome)] += 1
            # sum up features across trajectory (or choose final step only)
            for (_, feats) in traj:
                for oi, bit in feats.items():
                    feat_sums[int(oi)] = feat_sums.get(int(oi), 0) + int(bit)
        outcome_hist = {int(i): int(final_counts[i]) for i in range(self.K)}
        avg_feats = {oi: (feat_sums[oi] / (num_rollouts * horizon)) for oi in feat_sums.keys()}
        return {"outcome_counts": outcome_hist, "avg_feature_rate": avg_feats}

    def simulate_counterfactual(self,
                                start_outcome: int,
                                hypothesis: Dict[str, Any],
                                actions: List[Optional[int]],
                                horizon: int,
                                num_sims: int = 200) -> Dict[str, Any]:
        """
        Evaluate a hypothesis (a change to APRA order models or priors) by simulating many rollouts.

        hypothesis format (simple supported types):
         - {"type":"boost_outcome", "outcome":int, "order":int, "factor":1.2}
         - {"type":"suppress_outcome", "outcome":int, "order":int, "factor":0.8}
         - {"type":"increase_prior", "outcome":int, "delta":0.05}

        The function makes a shallow copy of APRA order_models and/or priors, applies the hypothesis,
        constructs a temporary WorldModel using the modified APRA object, and returns aggregated stats.
        """
        # try to clone apra object shallowly
        try:
            apra_copy = deepcopy(self.apra)
        except Exception:
            # best-effort: if deepcopy fails, we will mutate a minimal surrogate
            class _LightAPRA:
                pass
            apra_copy = _LightAPRA()
            # copy K and order_models if possible
            om = _get_order_models(self.apra)
            apra_copy.K = self.K
            apra_copy.order_models = {int(k): (v.copy() if isinstance(v, np.ndarray) else np.array(v, dtype=float)) for k, v in om.items()}
            # create minimal interface used by WorldModel simulate_step
            # note: won't support priors in sampling but adequate for feature sampling

        # apply hypothesis
        htype = hypothesis.get("type")
        if htype == "boost_outcome":
            o = int(hypothesis["outcome"]); oi = int(hypothesis["order"]); f = float(hypothesis.get("factor", 1.2))
            om = _get_order_models(apra_copy)
            if oi in om:
                vec = om[oi].copy()
                # increase probability for outcome o (scale)
                vec[o] = min(0.999, vec[o] * f)
                # renormalization across outcomes is not required (these are conditional probs P(feature|outcome))
                if hasattr(apra_copy, "order_models"):
                    apra_copy.order_models[oi] = vec
                else:
                    apra_copy.order_models = apra_copy.order_models if hasattr(apra_copy, "order_models") else {}
                    apra_copy.order_models[oi] = vec
        elif htype == "suppress_outcome":
            o = int(hypothesis["outcome"]); oi = int(hypothesis["order"]); f = float(hypothesis.get("factor", 0.8))
            om = _get_order_models(apra_copy)
            if oi in om:
                vec = om[oi].copy()
                vec[o] = max(1e-9, vec[o] * f)
                if hasattr(apra_copy, "order_models"):
                    apra_copy.order_models[oi] = vec
                else:
                    apra_copy.order_models = apra_copy.order_models if hasattr(apra_copy, "order_models") else {}
                    apra_copy.order_models[oi] = vec
        elif htype == "increase_prior":
            # best-effort: if apra_copy has set_priors function
            o = int(hypothesis["outcome"]); delta = float(hypothesis.get("delta", 0.05))
            if hasattr(apra_copy, "get_priors") and hasattr(apra_copy, "set_priors"):
                pri = list(apra_copy.get_priors())
                pri[o] = min(0.999, pri[o] + delta)
                # normalize
                s = sum(pri) + EPS
                pri = [p / s for p in pri]
                apra_copy.set_priors(pri)
            else:
                # if not supported, ignore
                pass
        else:
            # unsupported hypothesis -> no-op
            pass

        # create temp worldmodel with modified apra
        tmp_world = WorldModel(apra_copy, transition_matrix=self.transition, rng_seed=None, action_bias_fn=self.action_bias_fn)
        # run rollouts
        final_counts = np.zeros(self.K, dtype=int)
        order_counts: Dict[int, int] = {}
        for s in range(num_sims):
            actions_seq = [actions[i % len(actions)] for i in range(horizon)]
            traj = tmp_world.simulate_rollout(start_outcome, actions_seq, horizon)
            final_counts[int(traj[-1][0])] += 1
            for (_, feats) in traj:
                for oi, bit in feats.items():
                    order_counts[oi] = order_counts.get(oi, 0) + int(bit)
        return {
            "hypothesis": hypothesis,
            "final_outcome_hist": {int(i): int(final_counts[i]) for i in range(self.K)},
            "avg_feature_rate": {int(oi): (order_counts[oi] / (num_sims * horizon)) for oi in order_counts}
        }

# quick local demo
if __name__ == "__main__":
    # minimal demo using simple order_models
    from .apra_algorithm import APRAAlgorithm
    outcomes = ["win", "lose", "draw"]
    priors = [0.5, 0.25, 0.25]
    order_models = {0: [0.8, 0.2, 0.5], 1: [0.4, 0.6, 0.3]}
    apra = APRAAlgorithm(outcomes, priors, order_models, seed=123)
    wm = WorldModel(apra, rng_seed=7)
    start = 0
    actions = [0, 1, None]
    traj = wm.simulate_rollout(start, actions, horizon=5)
    print("traj:", traj)
    cf = wm.simulate_counterfactual(start, {"type":"boost_outcome", "outcome":0, "order":0, "factor":1.5}, actions, horizon=3, num_sims=200)
    print("counterfactual result:", cf)