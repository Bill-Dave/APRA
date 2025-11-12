# apra/planner.py
"""
Planner module for APRA kernel

Provides planning utilities that use the APRA engine and WorldModel to choose actions.
Supports:
 - simple greedy action selection (one-step expected-utility)
 - rollout-based search (sampled action-sequence evaluation)
 - a lightweight MCTS (UCT) implementation for short horizons

Design notes
- Works with APRAAlgorithm or APRAEngine wrappers.
- Uses WorldModel.simulate_rollout for forward simulation.
- Uses deterministic RNG from apra.utils.make_rng when rng_seed supplied.
- utilities: numpy array shape (A, K) where A=actions, K=outcomes.

API
---
Planner(apra, world_model, utilities, rng_seed=None)
 - plan_greedy(posterior) -> dict
 - plan_rollout_search(posterior, budget_actions, horizon, rollouts, beam_width) -> dict
 - plan_mcts(posterior, budget_actions, horizon, sims, c_puct) -> dict
 - eval_action_sequence(posterior, action_seq, horizon, rollouts) -> float
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

try:
    from .utils import make_rng, EPS
except Exception:
    # minimal fallback
    import os as _os
    import numpy as _np
    EPS = 1e-12
    def make_rng(seed: Optional[int] = None) -> _np.random.RandomState:
        if seed is None:
            seed = int.from_bytes(_os.urandom(4), "little") & 0xFFFFFFFF
        return _np.random.RandomState(seed)

# try importing compute_action_value_with_rollouts for reuse
try:
    from .voi import compute_action_value_with_rollouts
except Exception:
    compute_action_value_with_rollouts = None  # type: ignore

# We'll try to import WorldModel for type hints
try:
    from .world import WorldModel  # type: ignore
except Exception:
    WorldModel = Any  # type: ignore

# -------------------------
# Planner class
# -------------------------
class Planner:
    def __init__(self,
                 apra: Any,
                 world_model: WorldModel,
                 utilities: np.ndarray,
                 rng_seed: Optional[int] = None):
        """
        apra: APRAAlgorithm or APRAEngine-like object
        world_model: WorldModel instance (must implement simulate_rollout)
        utilities: numpy array shape (A, K)
        rng_seed: optional integer for deterministic behavior
        """
        self.apra = apra
        self.world = world_model
        self.utilities = np.asarray(utilities, dtype=float)
        self.A = int(self.utilities.shape[0])
        self.K = int(self.utilities.shape[1])
        self.rng = make_rng(rng_seed)

    # ---------- evaluation helpers ----------
    def expected_utility_action(self, posterior: np.ndarray, action_idx: int) -> float:
        """
        Compute immediate expected utility of choosing action_idx (one-step).
        """
        posterior = np.asarray(posterior, dtype=float)
        if action_idx < 0 or action_idx >= self.A:
            raise ValueError("action_idx out of range")
        eu = float(np.dot(self.utilities[action_idx, :], posterior))
        return eu

    def eval_action_sequence(self,
                             posterior: np.ndarray,
                             action_seq: List[int],
                             horizon: int,
                             rollouts: int = 200) -> float:
        """
        Monte-Carlo estimate of average cumulative utility for a fixed action sequence.
        Uses world_model.simulate_rollout starting from a latent outcome sampled from posterior.
        If compute_action_value_with_rollouts is available in voi.py, prefer that for single-action evaluation.
        """
        posterior = np.asarray(posterior, dtype=float)
        if compute_action_value_with_rollouts is None:
            # custom small implementation for full sequence
            total = 0.0
            for _ in range(rollouts):
                start = int(self.rng.choice(self.K, p=posterior))
                # repeat sequence if shorter than horizon
                actions = [action_seq[i % len(action_seq)] for i in range(horizon)]
                traj = self.world.simulate_rollout(start, actions, horizon)
                cum = 0.0
                for t, (outcome_t, feats) in enumerate(traj):
                    act = actions[t]
                    act_idx = int(act) if 0 <= int(act) < self.A else 0
                    cum += float(self.utilities[act_idx, int(outcome_t)])
                total += cum
            return float(total / max(1, rollouts))
        else:
            # if action_seq is single action, delegate to compute_action_value_with_rollouts
            if len(action_seq) == 1:
                return compute_action_value_with_rollouts(self.world, self.apra, posterior, int(action_seq[0]), self.utilities, horizon=horizon, rollouts=rollouts)
            # otherwise fallback to generic
            total = 0.0
            for _ in range(rollouts):
                start = int(self.rng.choice(self.K, p=posterior))
                actions = [action_seq[i % len(action_seq)] for i in range(horizon)]
                traj = self.world.simulate_rollout(start, actions, horizon)
                cum = 0.0
                for t, (outcome_t, feats) in enumerate(traj):
                    act = actions[t]
                    act_idx = int(act) if 0 <= int(act) < self.A else 0
                    cum += float(self.utilities[act_idx, int(outcome_t)])
                total += cum
            return float(total / max(1, rollouts))

    # ---------- greedy planner ----------
    def plan_greedy(self, posterior: np.ndarray) -> Dict[str, Any]:
        """
        Select the action with highest immediate expected utility.
        Returns dict {best_action, expected_value, eus}
        """
        posterior = np.asarray(posterior, dtype=float)
        eus = np.dot(self.utilities, posterior)  # (A,)
        best_idx = int(np.argmax(eus))
        return {"method": "greedy", "best_action": best_idx, "expected_value": float(eus[best_idx]), "eus": eus.tolist()}

    # ---------- rollout-based beam search ----------
    def plan_rollout_search(self,
                            posterior: np.ndarray,
                            budget_actions: List[int],
                            horizon: int = 3,
                            rollouts_per_seq: int = 150,
                            beam_width: int = 5,
                            expand_per_candidate: int = 3) -> Dict[str, Any]:
        """
        Beam-style rollout search:
         - Start with empty sequences.
         - Iteratively expand each candidate by appending each action in budget_actions (limited by expand_per_candidate randomness).
         - Evaluate newly formed sequences via eval_action_sequence (Monte Carlo).
         - Keep top beam_width sequences.
        Returns best sequence found and its estimated value.

        This method trades compute for robustness and is simple to implement on phone/Colab.
        """
        posterior = np.asarray(posterior, dtype=float)
        # normalize budget actions
        budget_actions = [int(a) for a in budget_actions]
        if len(budget_actions) == 0:
            raise ValueError("budget_actions must be non-empty")

        # candidate sequences: list of (seq, value_est)
        candidates: List[Tuple[List[int], float]] = [ ([], 0.0) ]
        for depth in range(horizon):
            new_candidates: List[Tuple[List[int], float]] = []
            for seq, _ in candidates:
                # choose expansion set: all budget actions or a random subset if too many
                actions_to_try = budget_actions if len(budget_actions) <= expand_per_candidate else list(self.rng.choice(budget_actions, size=expand_per_candidate, replace=False))
                for a in actions_to_try:
                    new_seq = seq + [int(a)]
                    val = self.eval_action_sequence(posterior, new_seq, horizon, rollouts=rollouts_per_seq)
                    new_candidates.append((new_seq, val))
            # select top beam_width
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[: max(1, beam_width)]
        # best candidate is first
        best_seq, best_val = candidates[0]
        # derive best immediate action (first of sequence) if any
        best_action = int(best_seq[0]) if len(best_seq) > 0 else int(np.argmax(np.dot(self.utilities, posterior)))
        return {"method": "rollout_search", "best_sequence": best_seq, "best_action": best_action, "expected_value": float(best_val), "candidates": [{"seq": c[0], "val": float(c[1])} for c in candidates]}

    # ---------- lightweight MCTS (UCT) ----------
    class _MCTSNode:
        def __init__(self, parent, action_from_parent: Optional[int], prior=0.0):
            self.parent = parent
            self.children: Dict[int, "Planner._MCTSNode"] = {}
            self.action_from_parent = action_from_parent
            self.visits = 0
            self.value = 0.0
            self.prior = prior  # optional prior for PUCT-style selection

    def plan_mcts(self,
                  posterior: np.ndarray,
                  budget_actions: List[int],
                  horizon: int = 3,
                  sims: int = 200,
                  c_puct: float = 1.0) -> Dict[str, Any]:
        """
        Lightweight MCTS using UCT/PUCT selection. Limited to shallow horizons (e.g. 2-4).
        Each simulation:
          - selection down the tree using UCT until leaf or depth == horizon
          - expansion: add children (all budget actions at node)
          - rollout: simulate random actions to horizon and return cumulative reward
          - backup: propagate reward up the tree

        Returns best action (child of root) and stats.
        """
        posterior = np.asarray(posterior, dtype=float)
        budget_actions = [int(a) for a in budget_actions]
        root = Planner._MCTSNode(self, action_from_parent=None)
        # create initial children (one-step)
        for a in budget_actions:
            root.children[a] = Planner._MCTSNode(root, action_from_parent=a)

        def uct_select(node: Planner._MCTSNode) -> Planner._MCTSNode:
            # UCT score = Q/N + c * sqrt(ln(parent.N)/N)
            best_score = -1e18
            best_child = None
            for a, ch in node.children.items():
                if ch.visits == 0:
                    score = 1e9 + ch.prior  # encourage unvisited
                else:
                    exploitation = ch.value / ch.visits
                    exploration = c_puct * math.sqrt(math.log(max(1, node.visits) + 1.0) / ch.visits)
                    score = exploitation + exploration + 0.01 * ch.prior
                if score > best_score:
                    best_score = score
                    best_child = ch
            assert best_child is not None
            return best_child

        def expand(node: Planner._MCTSNode):
            # add all budget actions as children (if not already present)
            for a in budget_actions:
                if a not in node.children:
                    node.children[a] = Planner._MCTSNode(node, action_from_parent=a)

        def rollout_from_node(node: Planner._MCTSNode) -> float:
            # derive partial action sequence from root to node
            seq: List[int] = []
            cur = node
            # walk up to root
            while cur is not None and cur.action_from_parent is not None:
                seq.append(cur.action_from_parent)
                cur = cur.parent
            seq = list(reversed(seq))
            # if seq empty, pick random first action
            if len(seq) == 0:
                seq = [int(self.rng.choice(budget_actions))]
            # complete with random actions up to horizon
            while len(seq) < horizon:
                seq.append(int(self.rng.choice(budget_actions)))
            # evaluate sequence
            val = self.eval_action_sequence(posterior, seq, horizon, rollouts=50)
            return val

        for sim in range(max(1, sims)):
            node = root
            depth = 0
            # selection
            while depth < horizon:
                if len(node.children) == 0:
                    expand(node)
                # choose child with UCT
                node = uct_select(node)
                depth += 1
                # if leaf (no grandchildren yet) break for expansion
                # we continue until depth==horizon for simplicity
            # expansion at leaf
            expand(node)
            # rollout
            reward = rollout_from_node(node)
            # backup
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value += reward
                cur = cur.parent

        # choose best child of root by average value
        best_val = -1e18
        best_action = None
        stats = []
        for a, ch in root.children.items():
            avg = (ch.value / ch.visits) if ch.visits > 0 else -1e18
            stats.append({"action": int(a), "visits": int(ch.visits), "avg_value": float(avg)})
            if avg > best_val:
                best_val = avg
                best_action = a
        return {"method": "mcts", "best_action": int(best_action) if best_action is not None else 0, "expected_value": float(best_val), "stats": stats}

# -------------------------
# Quick demo / smoke test
# -------------------------
if __name__ == "__main__":
    # minimal smoke test using APRAAlgorithm / WorldModel if available
    try:
        from .apra_algorithm import APRAAlgorithm  # type: ignore
        from .world import WorldModel  # type: ignore
        outcomes = ["win", "lose", "draw"]
        priors = [0.5, 0.25, 0.25]
        order_models = {0: [0.8, 0.2, 0.5], 1: [0.4, 0.6, 0.3]}
        apra = APRAAlgorithm(outcomes, priors, order_models, seed=123)
        wm = WorldModel(apra, rng_seed=7)
        utilities = np.array([
            [10.0, -100.0, 5.0],  # action 0
            [2.0, 1.0, 3.0],      # action 1
            [5.0, -10.0, 2.0]     # action 2
        ], dtype=float)
        planner = Planner(apra, wm, utilities, rng_seed=42)
        post = apra.compute_posterior_from_prefix([(0, 1)])
        print("Greedy plan:", planner.plan_greedy(post))
        print("Rollout search:", planner.plan_rollout_search(post, budget_actions=[0,1,2], horizon=3, rollouts_per_seq=80, beam_width=3))
        print("MCTS:", planner.plan_mcts(post, budget_actions=[0,1,2], horizon=3, sims=120))
    except Exception as e:
        print("Planner self-test skipped or errored:", e)
```0