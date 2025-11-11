# apra/utils.py
"""
Utilities for the APRA kernel.

Contents:
 - deterministic RNG / seeding helpers
 - numeric helpers (stable softmax, logsumexp wrapper)
 - lightweight logging helper
 - small serialization helpers
 - simple type aliases
"""

from __future__ import annotations
import math
import json
import os
import sys
import typing as _t
from typing import Any, Dict, Iterable, Sequence, Tuple, List, Optional
import numpy as np

# Types
Array = np.ndarray
Number = _t.Union[int, float]

EPS = 1e-12

# ---------- numeric helpers ----------
def stable_softmax_vec(log_scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1-D numpy array."""
    if not isinstance(log_scores, np.ndarray):
        log_scores = np.array(log_scores, dtype=float)
    m = np.max(log_scores)
    ex = np.exp(log_scores - m)
    denom = ex.sum() + EPS
    return ex / denom

def logsumexp(logv: np.ndarray) -> float:
    """Compute log-sum-exp of a 1-D numpy vector."""
    if not isinstance(logv, np.ndarray):
        logv = np.array(logv, dtype=float)
    m = np.max(logv)
    return float(m + math.log(float(np.sum(np.exp(logv - m))) + EPS))

def clamp_probs(arr: _t.Sequence[float], floor: float = 1e-9, ceil: float = 1.0-1e-9) -> np.ndarray:
    """Clip a sequence of floats into (floor, ceil) and return numpy array normalized if it sums != 1."""
    a = np.clip(np.array(arr, dtype=float), floor, ceil)
    s = a.sum()
    if s <= 0:
        # fallback to uniform
        a = np.ones_like(a) / len(a)
    else:
        a = a / (s + EPS)
    return a

# ---------- seeding / RNG ----------
def make_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Create a deterministic RandomState given an integer seed.
    If seed is None, tries to read APRA_SEED env var, else uses os.urandom.
    """
    if seed is None:
        senv = os.environ.get("APRA_SEED", None)
        if senv is not None:
            try:
                seed = int(senv)
            except Exception:
                seed = None
    if seed is None:
        # fallback from os.urandom -> stable 32-bit int
        seed = int.from_bytes(os.urandom(4), "little") & 0xFFFFFFFF
    return np.random.RandomState(seed)

def rng_choice(rng: np.random.RandomState, a: Sequence[Any], p: Optional[Sequence[float]] = None) -> Any:
    """Deterministic choice wrapper using provided RandomState."""
    if p is None:
        idx = rng.randint(0, len(a))
        return a[idx]
    else:
        p = np.array(p, dtype=float)
        p = p / (p.sum() + EPS)
        idx = int(rng.choice(len(a), p=p))
        return a[idx]

# ---------- serialization ----------
def to_json_safe(obj: Any) -> str:
    """Simple wrapper to JSON-dump objects that may contain numpy types."""
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")
    return json.dumps(obj, default=_default, indent=2)

def from_json_safe(s: str) -> Any:
    return json.loads(s)

# ---------- logging helper ----------
def debug_log(msg: str, level: str = "info", verbose_env: str = "APRA_VERBOSE") -> None:
    """
    lightweight logger. Controlled by APRA_VERBOSE env var:
     - if unset or 0: prints only warnings/errors
     - if 1: prints info
     - if >=2: prints debug
    """
    try:
        vv = int(os.environ.get(verbose_env, "0"))
    except Exception:
        vv = 0
    level = (level or "info").lower()
    rank = {"error":0, "warn":0, "warning":0, "info":1, "debug":2}.get(level, 1)
    if vv >= rank:
        print(f"[apra-utils:{level}] {msg}", file=sys.stdout if rank>0 else sys.stderr)

# ---------- small helpers ----------
def ensure_list(x: Optional[_t.Iterable[Any]]) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x)

def canonicalize_outcomes(outcomes: Sequence[str]) -> List[str]:
    return [str(o) for o in outcomes]

def vec_to_probs(vec: Sequence[Number]) -> np.ndarray:
    arr = np.array(vec, dtype=float)
    arr = np.clip(arr, 1e-12, None)
    s = arr.sum()
    if s <= 0:
        arr = np.ones_like(arr) / arr.shape[0]
    else:
        arr = arr / (s + EPS)
    return arr

# ---------- small sanity test ----------
def _self_test():
    rng = make_rng(42)
    v = np.array([1.0, 2.0, 3.0])
    sm = stable_softmax_vec(v)
    print("softmax:", sm, "sum:", sm.sum())
    print("clamp_probs:", clamp_probs([0.0, 0.0, 0.0]))
    print("json safe:", to_json_safe({"a": np.array([1,2,3])}))
    debug_log("this is debug (show when APRA_VERBOSE>=2)", level="debug")
    debug_log("this is info (show when APRA_VERBOSE>=1)", level="info")
if __name__ == "__main__":
    _self_test()