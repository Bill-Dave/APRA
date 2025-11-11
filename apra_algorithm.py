# apra/apra_algorithm.py 
"""
APRAAlgorithm - an implementation of the Almost-Perfect Randomness Algorithm (APRA)
Implements:
 - multi-order probabilistic model: P(features | outcome) with explicit order indices
 - inter-order influence matrix W that lets observed orders modulate other orders' feature probabilities
 - iterative posterior refinement to account for higher-order effects (approximate marginalization)
 - deterministic sampling utilities and serialization

Usage:
    from apra.apra_algorithm import APRAAlgorithm
    apra = APRAAlgorithm(outcomes, priors, order_models, order_interaction=W)
    posterior = apra.compute_posterior_from_prefix(observed_prefix)
    # posterior is a numpy vector over outcomes

Design notes:
 - order_models: dict order_idx -> P(feature=1 | outcome) vector (length K)
 - order_interaction: numpy (M x M) matrix where M = number of orders; entry w_ji means observed order j
    will shift the log-odds of order i by w_ji * (observed_j - expected_j)
 - refinement iterations: we iteratively adjust order likelihoods given the current posterior,
    capturing higher-order cascades approximately.
"""

from __future__ import annotations
import math
import json
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os

EPS = 1e-12

def stable_softmax(log_scores: np.ndarray) -> np.ndarray:
    m = np.max(log_scores)
    ex = np.exp(log_scores - m)
    return ex / (ex.sum() + EPS)

def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0-1e-9)
    return math.log(p / (1.0 - p))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

class APRAAlgorithm:
    """
    APRAAlgorithm: APRA-native engine implementing multi-order randomness with interactions.

    Constructor:
        outcomes: List[str] K outcomes
        priors: List[float] length K
        order_models: Dict[int, List[float]] mapping order -> P(feature=1 | outcome) (length K)
        order_interaction: Optional[np.ndarray] shape (M, M) where M = number of orders present.
            w[j,i] means observed order j influences order i (note ordering).
        seed: Optional int for deterministic sampling
    """
    def __init__(self,
                 outcomes: List[str],
                 priors: List[float],
                 order_models: Optional[Dict[int, List[float]]] = None,
                 order_interaction: Optional[np.ndarray] = None,
                 normalize_priors: bool = True,
                 seed: Optional[int] = None):
        if len(outcomes) == 0:
            raise ValueError("outcomes must be non-empty")
        self.outcomes = list(outcomes)
        self.K = len(self.outcomes)

        p = np.array(priors, dtype=float)
        if p.shape[0] != self.K:
            raise ValueError("priors length must equal K")
        if normalize_priors:
            p = p / (p.sum() + EPS)
        self.log_prior = np.log(p + EPS)

        # order models: map order_index -> numpy array shape (K,)
        self.order_models: Dict[int, np.ndarray] = {}
        self.order_index_list: List[int] = []
        if order_models:
            for k, vec in order_models.items():
                arr = np.array(vec, dtype=float)
                if arr.shape[0] != self.K:
                    raise ValueError("order model length mismatch")
                self.order_models[int(k)] = np.clip(arr, 1e-9, 1.0 - 1e-9)
            self.order_index_list = sorted(list(self.order_models.keys()))
        # interaction matrix W: shape (M, M) where M = number of orders
        if order_interaction is not None:
            W = np.array(order_interaction, dtype=float)
            m = W.shape[0]
            if W.shape[1] != m:
                raise ValueError("order_interaction must be square (M x M)")
            # if M doesn't match number of orders, we will map by index order
            self.W = W.copy()
        else:
            # default zero interactions
            m = len(self.order_index_list)
            self.W = np.zeros((m, m), dtype=float)

        # mapping order_idx -> position in W
        self.order_pos: Dict[int, int] = {order: i for i, order in enumerate(self.order_index_list)}

        # RNG
        self._rng = np.random.RandomState(seed if seed is not None else (int.from_bytes(os.urandom(4), 'little') & 0xffffffff))

    # ---------------------------
    # core: compute posterior with APRA refinements
    # ---------------------------
    def compute_posterior_from_prefix(self,
                                      observed_prefix: List[Tuple[int, int]],
                                      refinement_iters: int = 4,
                                      verbose: bool = False) -> np.ndarray:
        """
        Main APRA posterior calculation.
        Steps:
         1. Start with log_prior
         2. For each observed (order,bit), compute base order likelihood log P(bit | outcome)
         3. Iteratively adjust order likelihoods using interaction matrix W and current posterior:
            - For order i, effective logit = logit(base_p_i[outcome]) + sum_j w[j,i] * (obs_j - expected_j)
            - Recompute likelihoods and posterior; repeat refinement for a few iterations
         4. Return final posterior (prob over outcomes)
        """
        # Normalize observed_prefix into dict order->bit (if multiple, last wins)
        obs_map = {int(o): int(b) for (o, b) in observed_prefix}

        # initialize log_scores from prior
        log_scores = self.log_prior.copy()

        # prepare base log-likelihood per order as matrix shape (M, K)
        M = len(self.order_index_list)
        if M == 0:
            # no order models available -> return prior
            return stable_softmax(log_scores)

        base_p = np.ones((M, self.K), dtype=float)
        for i, order in enumerate(self.order_index_list):
            base_p[i, :] = self.order_models.get(order, np.ones(self.K) * 0.5)
        # convert to logit-space per outcome to allow additive interactions
        base_logit = np.vectorize(lambda x: logit(x))(base_p)  # shape (M, K)

        # initial posterior (we will refine)
        posterior = np.exp(log_scores - np.max(log_scores))
        posterior = posterior / (posterior.sum() + EPS)

        # build observed vector obs_j for each order j; if unknown, use expected P(feature=1)
        obs_vec = np.zeros(M, dtype=float)
        expected_vec = np.zeros(M, dtype=float)
        for j, order in enumerate(self.order_index_list):
            if order in obs_map:
                obs_vec[j] = float(obs_map[order])
            else:
                # expected value under current posterior: sum_outcome P(outcome) * P(feature=1|outcome)
                expected_vec[j] = float(np.sum(posterior * base_p[j, :]))
                obs_vec[j] = float(expected_vec[j])

        # Iterative refinement: update log_scores by adding per-order adjusted log-likelihoods
        for it in range(refinement_iters):
            # compute adjustments for each order i given observed/expected
            # delta_logit_i(outcome) = sum_j W[j,i] * (obs_j - expected_j)
            delta = np.zeros((M, ), dtype=float)
            if M > 0:
                # compute residual r_j = obs_j - expected_j
                r = obs_vec - expected_vec
                # delta per order i
                delta = np.dot(r, self.W)  # shape (M,)
            # build total log-likelihood per outcome: sum over i log P_i(obs_i | outcome) with adjusted logits
            total_log_like = np.zeros(self.K, dtype=float)
            for i in range(M):
                # for this order i, for each outcome k compute adjusted p = sigmoid(base_logit[i,k] + delta[i])
                adjusted_logit = base_logit[i, :] + delta[i]
                adjusted_p = 1.0 / (1.0 + np.exp(-adjusted_logit))
                # now if we have actual observed bit for this order, use that; else use expected (treated as soft evidence)
                order_idx = self.order_index_list[i]
                if order_idx in obs_map:
                    b = obs_map[order_idx]
                    # log P(b | outcome)
                    log_pf = np.log(adjusted_p + EPS) if b == 1 else np.log(1.0 - adjusted_p + EPS)
                else:
                    # if not observed, use marginal likelihood (mixture) approximate as: P(obs ~ expected) ~ adjusted_p^expected * (1-adjusted_p)^(1-expected)
                    # here expected_vec[i] is expectation under previous posterior; treat it as fractional evidence
                    q = expected_vec[i]
                    log_pf = q * np.log(adjusted_p + EPS) + (1.0 - q) * np.log(1.0 - adjusted_p + EPS)
                total_log_like += log_pf
            # combine with prior and normalize
            log_scores = self.log_prior + total_log_like
            posterior = stable_softmax(log_scores)
            # recompute expected_vec under new posterior
            for i in range(M):
                expected_vec[i] = float(np.sum(posterior * base_p[i, :]))
                # if order not observed, obs_vec is updated to expected (soft)
                order_idx = self.order_index_list[i]
                if order_idx not in obs_map:
                    obs_vec[i] = expected_vec[i]
            if verbose:
                print(f"[APRA] iter={it} posterior={posterior.tolist()} delta={delta.tolist()}")
        return posterior

    # ---------------------------
    # sampling helpers
    # ---------------------------
    def sample_true_outcome(self) -> int:
        p = np.exp(self.log_prior - np.max(self.log_prior))
        p = p / (p.sum() + EPS)
        return int(self._rng.choice(len(p), p=p))

    def sample_observed_features_for_outcome(self, outcome_idx: int, order_idxs: Optional[List[int]] = None) -> Dict[int, int]:
        """
        Given an outcome, sample observed bits for specified orders.
        Returns dict order -> bit
        """
        res = {}
        if order_idxs is None:
            order_idxs = self.order_index_list[:]
        for order in order_idxs:
            i = self.order_pos.get(order, None)
            if i is None:
                # unknown order -> sample fair
                res[order] = 1 if self._rng.rand() < 0.5 else 0
            else:
                p = float(self.order_models.get(order, np.ones(self.K) * 0.5)[int(outcome_idx)])
                res[order] = 1 if self._rng.rand() < p else 0
        return res

    # ---------------------------
    # helpers: mutate interaction matrix, add orders, serialization
    # ---------------------------
    def set_order_interaction(self, W: np.ndarray):
        W = np.array(W, dtype=float)
        if W.shape[0] != W.shape[1]:
            raise ValueError("W must be square")
        self.W = W.copy()
        # if we have new shape mismatch vs existing order list, user must rebuild mapping
        if W.shape[0] != len(self.order_index_list):
            # silent: we will not remap; user should reshape accordingly
            pass

    def add_order_model(self, order_idx: int, probs: List[float]):
        arr = np.array(probs, dtype=float)
        if arr.shape[0] != self.K:
            raise ValueError("order model length mismatch")
        if order_idx in self.order_models:
            self.order_models[order_idx] = np.clip(arr, 1e-9, 1.0-1e-9)
        else:
            self.order_models[order_idx] = np.clip(arr, 1e-9, 1.0-1e-9)
            self.order_index_list = sorted(self.order_models.keys())
            self.order_pos = {order: i for i, order in enumerate(self.order_index_list)}
            # grow W as zero padding if needed
            m = len(self.order_index_list)
            if getattr(self, "W", None) is None:
                self.W = np.zeros((m, m), dtype=float)
            elif self.W.shape[0] < m:
                W2 = np.zeros((m, m), dtype=float)
                W2[:self.W.shape[0], :self.W.shape[1]] = self.W
                self.W = W2

    def to_json(self) -> str:
        state = {
            "outcomes": self.outcomes,
            "log_prior": self.log_prior.tolist(),
            "order_models": {int(k): v.tolist() for k, v in self.order_models.items()},
            "order_index_list": self.order_index_list,
            "W": self.W.tolist()
        }
        return json.dumps(state)

    @staticmethod
    def from_json(s: str) -> "APRAAlgorithm":
        j = json.loads(s)
        obj = APRAAlgorithm(j["outcomes"], [math.exp(x) for x in j["log_prior"]], j["order_models"], np.array(j.get("W", [])))
        return obj

    def copy(self) -> "APRAAlgorithm":
        return deepcopy(self)

# simple demo when run as script
if __name__ == "__main__":
    outcomes = ["win", "lose", "draw"]
    priors = [0.5, 0.25, 0.25]
    order_models = {
        0: [0.8, 0.2, 0.5],
        1: [0.4, 0.6, 0.3],
        2: [0.3, 0.3, 0.9]
    }
    # small interaction: order 0 pushes order1 positively, order1 pushes order2 negatively
    W = np.array([
        [0.0, 0.25, 0.0],
        [0.0, 0.0, -0.20],
        [0.0, 0.0, 0.0]
    ])
    ap = APRAAlgorithm(outcomes, priors, order_models, order_interaction=W, seed=42)
    # observe order 0 = 1
    obs = [(0, 1)]
    post = ap.compute_posterior_from_prefix(obs, refinement_iters=6, verbose=True)
    print("posterior:", post.tolist())