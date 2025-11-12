# apra/meta_reasoner.py
"""
Meta-reasoner for APRA kernel

Provides:
 - generation of candidate hypotheses (counterfactuals) from a posterior
 - scoring hypotheses by estimating expected improvement in decision utility
 - simple simulation of counterfactuals using APRA order models and optional WorldModel

API:
    mr = MetaReasoner(apra, world_model=None)
    hypos = mr.generate_hypotheses(posterior, top_k=3)
    score = mr.score_hypothesis(hypos[0], posterior, utilities, num_sims=128, horizon=3)

Notes:
 - Works with APRAAlgorithm or the APRAEngine wrapper (compatibility layer).
 - Uses Monte-Carlo sampling of outcomes and order-features to estimate hypothesis effect.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import random

# Try to import WorldModel for optional counterfactual sim support
try:
    from .world import WorldModel  # type: ignore
except Exception:
    WorldModel = None  # type: ignore

EPS = 1e-12


def _get_order_models(apra: Any) -> Dict[int, np.ndarray]:
    """Return a dict order_idx -> numpy array (K,) of P(feature=1|outcome)."""
    if hasattr(apra, "order_models"):
        om = getattr(apra, "order_models")
        return {int(k): (v if isinstance(v, np.ndarray) else np.array(v, dtype=float)) for k, v in om.items()}
    if hasattr(apra, "_apra") and hasattr(apra._apra, "order_models"):
        om = getattr(apra._apra, "order_models")
        return {int(k): (v if isinstance(v, np.ndarray) else np.array(v, dtype=float)) for k, v in om.items()}
    return {}


def _get_priors(apra: Any) -> np.ndarray:
    """Return prior probabilities vector (K,) from apra or its wrapper."""
    try:
        if hasattr(apra, "get_priors"):
            p = np.asarray(apra.get_priors(), dtype=float)
            return p / (p.sum() + EPS)
    except Exception:
        pass
    # try compute posterior with empty evidence
    try:
        p = apra.compute_posterior_from_prefix([])  # type: ignore
        return np.asarray(p, dtype=float) / (np.sum(p) + EPS)
    except Exception:
        pass
    # fallback: uniform if K detectable
    om = _get_order_models(apra)
    if len(om) > 0:
        K = len(next(iter(om.values())))
        return np.ones(K, dtype=float) / float(K)
    raise RuntimeError("Unable to extract priors from apra object")


class MetaReasoner:
    """
    MetaReasoner manages hypothesis generation and scoring.

    Parameters
    ----------
    apra : APRAAlgorithm or APRAEngine-compatible object
    world_model : optional WorldModel for realistic sampling (falls back to apra order_models)
    rng_seed : optional seed for deterministic behavior
    """

    def __init__(self, apra: Any, world_model: Optional[Any] = None, rng_seed: Optional[int] = None):
        self.apra = apra
        self.world = world_model
        self.rng = np.random.RandomState(rng_seed if rng_seed is not None else np.random.randint(1 << 30))

    def generate_hypotheses(self, posterior: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Generate a small set of plausible hypotheses focused on the top-k outcomes.

        Each hypothesis is a dict describing a counterfactual change, example forms:
            {'type':'boost_outcome', 'outcome': int, 'order': int, 'factor': 1.2}
            {'type':'suppress_outcome', 'outcome': int, 'order': int, 'factor': 0.8}
            {'type':'increase_prior', 'outcome': int, 'delta': 0.05}
            {'type':'set_order_prob', 'order':int, 'outcome':int, 'new_p':float}

        The generator inspects available orders and suggests modifications near them.
        """
        posterior = np.asarray(posterior, dtype=float)
        K = posterior.shape[0]
        idxs = list(np.argsort(-posterior)[:max(1, min(top_k, K))])
        hypos: List[Dict[str, Any]] = []
        order_models = _get_order_models(self.apra)
        order_keys = sorted(list(order_models.keys()))
        for o in idxs:
            # small prior bump hypothesis
            hypos.append({'type': 'increase_prior', 'outcome': int(o), 'delta': 0.03})
            # if orders exist, propose boost/suppress on a couple of orders
            for oi in order_keys[: min(3, len(order_keys))]:
                hypos.append({'type': 'boost_outcome', 'outcome': int(o), 'order': int(oi), 'factor': 1.2})
                hypos.append({'type': 'suppress_outcome', 'outcome': int(o), 'order': int(oi), 'factor': 0.75})
                # propose setting the order prob directly toward a target
                base_p = float(order_models[oi][o])
                new_p = min(0.999, base_p * 1.25)
                hypos.append({'type': 'set_order_prob', 'order': int(oi), 'outcome': int(o), 'new_p': float(new_p)})
        # de-duplicate by string
        seen = set()
        uniq = []
        for h in hypos:
            k = jsonable(h)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(h)
        return uniq[: max(1, top_k * 3)]

    def _apply_hypothesis_to_copy(self, apra_copy: Any, hypothesis: Dict[str, Any]) -> None:
        """
        Mutate apra_copy in-place to reflect the hypothesis.
        Works best if apra_copy is a deep copy of the original APRA object.
        """
        htype = hypothesis.get('type')
        if htype == 'boost_outcome':
            o = int(hypothesis['outcome']); oi = int(hypothesis['order']); f = float(hypothesis.get('factor', 1.2))
            om = _get_order_models(apra_copy)
            if oi in om:
                vec = om[oi].copy()
                vec[o] = min(0.999, vec[o] * f)
                # attempt set: APRAAlgorithm has order_models dict; wrapper also exposes add_order_model
                try:
                    if hasattr(apra_copy, 'order_models'):
                        apra_copy.order_models[oi] = vec
                    elif hasattr(apra_copy, '_apra'):
                        apra_copy._apra.order_models[oi] = vec
                except Exception:
                    pass
        elif htype == 'suppress_outcome':
            o = int(hypothesis['outcome']); oi = int(hypothesis['order']); f = float(hypothesis.get('factor', 0.8))
            om = _get_order_models(apra_copy)
            if oi in om:
                vec = om[oi].copy()
                vec[o] = max(1e-9, vec[o] * f)
                try:
                    if hasattr(apra_copy, 'order_models'):
                        apra_copy.order_models[oi] = vec
                    elif hasattr(apra_copy, '_apra'):
                        apra_copy._apra.order_models[oi] = vec
                except Exception:
                    pass
        elif htype == 'increase_prior':
            o = int(hypothesis['outcome']); delta = float(hypothesis.get('delta', 0.05))
            # try to update log_prior or call set_priors
            try:
                if hasattr(apra_copy, 'get_priors') and hasattr(apra_copy, 'set_priors'):
                    pri = list(apra_copy.get_priors())
                    pri[o] = min(0.999, pri[o] + delta)
                    s = sum(pri) + EPS
                    pri = [p / s for p in pri]
                    apra_copy.set_priors(pri)
                    return
            except Exception:
                pass
            # fallback: try raw attribute
            try:
                if hasattr(apra_copy, '_apra') and hasattr(apra_copy._apra, 'log_prior'):
                    p = np.exp(apra_copy._apra.log_prior - np.max(apra_copy._apra.log_prior))
                    p = p / (p.sum() + EPS)
                    p[o] = min(0.999, p[o] + delta)
                    p = p / (p.sum() + EPS)
                    apra_copy._apra.log_prior = np.log(p + EPS)
            except Exception:
                pass
        elif htype == 'set_order_prob':
            oi = int(hypothesis['order']); o = int(hypothesis['outcome']); newp = float(hypothesis['new_p'])
            om = _get_order_models(apra_copy)
            if oi in om:
                vec = om[oi].copy()
                vec[o] = min(0.999, max(1e-9, newp))
                try:
                    if hasattr(apra_copy, 'order_models'):
                        apra_copy.order_models[oi] = vec
                    elif hasattr(apra_copy, '_apra'):
                        apra_copy._apra.order_models[oi] = vec
                except Exception:
                    pass
        else:
            # unsupported hypothesis type -> no-op
            pass

    def score_hypothesis(self,
                         hypothesis: Dict[str, Any],
                         posterior: np.ndarray,
                         utilities: np.ndarray,
                         num_sims: int = 128,
                         horizon: int = 3) -> float:
        """
        Estimate expected improvement (in utility) produced by applying hypothesis.

        Procedure (Monte-Carlo approx):
          - baseline_best = max_a E_u[a] with current posterior
          - For s = 1..num_sims:
              * sample an outcome a0 ~ posterior
              * create apra_copy and apply hypothesis to apra_copy
              * if hypothesis refers to an order, simulate observing that order's feature for outcome a0:
                    v = Bernoulli(p = apra_copy.order_models[order][a0])
                    compute posterior_v by updating current posterior with (order, v)
                else:
                    posterior_v = posterior_after_applying_prior_change (apra_copy prior) -- approximate by apra_copy prior
              * compute best_after = max_a utilities[a] . posterior_v
          - expected_improve = average(best_after - baseline_best)

        Returns expected_improve (float).
        """
        posterior = np.asarray(posterior, dtype=float)
        utilities = np.asarray(utilities, dtype=float)
        K = int(posterior.shape[0])
        baseline_eus = utilities.dot(posterior)
        baseline_best = float(np.max(baseline_eus))

        total_improve = 0.0
        order_models_orig = _get_order_models(self.apra)
        order_list = sorted(list(order_models_orig.keys()))

        for s in range(max(1, num_sims)):
            # sample an outcome from current posterior (latent ground truth for this sim)
            a0 = int(self.rng.choice(np.arange(K), p=posterior))
            # deep-copy apra to mutate safely
            try:
                apra_copy = deepcopy(self.apra)
            except Exception:
                # best-effort: shallow copy via JSON roundtrip if available
                try:
                    j = self.apra.to_json()
                    # attempt to import APRAAlgorithm or APRAEngine to recreate
                    from .apra_algorithm import APRAAlgorithm  # type: ignore
                    apra_copy = APRAAlgorithm.from_json(j)
                except Exception:
                    # if nothing works, skip simulation
                    continue

            # apply hypothesis to copy
            self._apply_hypothesis_to_copy(apra_copy, hypothesis)

            # compute posterior after hypothetical observation
            if 'order' in hypothesis:
                oi = int(hypothesis['order'])
                # find p_vec on the mutated apra_copy
                try:
                    if hasattr(apra_copy, "order_models"):
                        pvec = np.asarray(apra_copy.order_models[oi], dtype=float)
                    elif hasattr(apra_copy, "_apra") and hasattr(apra_copy._apra, "order_models"):
                        pvec = np.asarray(apra_copy._apra.order_models[oi], dtype=float)
                    else:
                        pvec = order_models_orig.get(oi, np.ones(K) * 0.5)
                except Exception:
                    pvec = order_models_orig.get(oi, np.ones(K) * 0.5)
                # sample observed bit for this outcome a0 with mutated pvec
                v = 1 if random.random() < float(pvec[a0]) else 0
                # compute posterior_v using Bayes: P(o|obs) proportional to P(o)*P(obs|o)
                # start with current posterior (reflects evidence so far)
                log_scores = np.log(posterior + EPS)
                if v == 1:
                    log_pf = np.log(pvec + EPS)
                else:
                    log_pf = np.log(1.0 - pvec + EPS)
                log_scores = log_scores + log_pf
                # normalize
                maxls = np.max(log_scores)
                post_v = np.exp(log_scores - maxls)
                post_v = post_v / (post_v.sum() + EPS)
            else:
                # no order in hypothesis: assume prior shift (increase_prior)
                # attempt to read new priors from apra_copy
                try:
                    post_v = _get_priors(apra_copy)
                except Exception:
                    post_v = posterior.copy()

            # compute best_after
            eus_after = utilities.dot(post_v)
            best_after = float(np.max(eus_after))
            total_improve += (best_after - baseline_best)

        expected_improve = float(total_improve / max(1, num_sims))
        return expected_improve

    def simulate_counterfactual(self,
                                hypothesis: Dict[str, Any],
                                start_outcome: int,
                                actions: List[int],
                                horizon: int = 3,
                                num_sims: int = 200) -> Dict[str, Any]:
        """
        Use world_model.simulate_counterfactual if available; otherwise emulate by
        applying hypothesis to a copy of APRA and using WorldModel constructed from it.
        Returns aggregated result dict.
        """
        if self.world is not None and hasattr(self.world, "simulate_counterfactual"):
            try:
                return self.world.simulate_counterfactual(start_outcome, hypothesis, actions, horizon, num_sims)
            except Exception:
                pass
        # fallback: build a temporary WorldModel using a mutated APRA copy and call its simulate_counterfactual
        try:
            apra_copy = deepcopy(self.apra)
        except Exception:
            # fallback simple shallow copy attempt via to_json/from_json
            try:
                j = self.apra.to_json()
                from .apra_algorithm import APRAAlgorithm  # type: ignore
                apra_copy = APRAAlgorithm.from_json(j)
            except Exception:
                apra_copy = None

        if apra_copy is None:
            return {"error": "unable_to_copy_apra"}

        # apply hypothesis
        self._apply_hypothesis_to_copy(apra_copy, hypothesis)

        # construct a temporary world model (best-effort)
        if WorldModel is None:
            return {"error": "WorldModel unavailable"}
        tmp_world = WorldModel(apra_copy, transition_matrix=getattr(self.world, "transition", None) if self.world is not None else None, rng_seed=None, action_bias_fn=getattr(self.world, "action_bias_fn", None) if self.world is not None else None)
        res = tmp_world.simulate_counterfactual(start_outcome, hypothesis, actions, horizon, num_sims=num_sims)
        return res


# tiny helper to create jsonable key for dedupe
def jsonable(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, sort_keys=True, default=lambda o: str(o))
    except Exception:
        return str(obj)


# Self-test quick smoke when run directly
if __name__ == "__main__":
    # minimal smoke using APRAAlgorithm (if available)
    try:
        from .apra_algorithm import APRAAlgorithm  # type: ignore
        outcomes = ["win", "lose", "draw"]
        priors = [0.5, 0.25, 0.25]
        order_models = {0: [0.8, 0.2, 0.5], 1: [0.4, 0.6, 0.3]}
        apra = APRAAlgorithm(outcomes, priors, order_models, seed=123)
        mr = MetaReasoner(apra)
        post = apra.compute_posterior_from_prefix([(0, 1)])
        utilities = np.array([[10, -100, 5], [2, 1, 3], [5, -10, 2]], dtype=float)
        hypos = mr.generate_hypotheses(post, top_k=2)
        print("Hypos:", hypos)
        sc = mr.score_hypothesis(hypos[0], post, utilities, num_sims=64, horizon=2)
        print("Hypothesis score:", sc)
    except Exception as e:
        print("Self-test skipped due to:", e)
```0