# learning/encoding/embedding_router.py
"""
Embedding router: abstracts over multiple embedding backends.

Supported providers (via ENV or explicit provider arg):
- 'mock'         -> deterministic pseudo-random vector from sha256
- 'sentence-transformers' (local) -> if sentence_transformers is installed
- 'openai'       -> OpenAI embeddings via OpenAI API
- 'hf-inference' -> Hugging Face Inference API (if HF token provided)

Functions:
- embed_texts(texts: List[str], provider: Optional[str]=None, model: Optional[str]=None, batch_size=16) -> List[List[float]]
"""
from typing import List, Optional
import os
import logging
import hashlib
import math

logger = logging.getLogger(__name__)

# optional imports guarded
try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import openai  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

import requests

DEFAULT_DIM = int(os.getenv("EMBED_DIM", "1536"))
DEFAULT_PROVIDER = os.getenv("EMBED_PROVIDER", "mock")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
HF_INFERENCE_API = os.getenv("HF_INFERENCE_API", None)


def _mock_vector_for_text(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    """
    Deterministic pseudo-random vector based on sha256 of text.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # seed from first 8 bytes
    seed = int.from_bytes(h[:8], "big")
    # produce reproducible floats in [-1,1]
    if _HAS_NUMPY:
        rng = np.random.RandomState(seed % (2 ** 32))
        vec = rng.rand(dim).astype(float) * 2.0 - 1.0
        return vec.tolist()
    else:
        # pure python fallback
        vals = []
        s = seed
        for i in range(dim):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            vals.append(((s % 10000) / 5000.0) - 1.0)
        return vals


def _embed_with_sentence_transformers(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed")
    m = SentenceTransformer(model_name)
    vecs = m.encode(texts, show_progress_bar=False)
    if _HAS_NUMPY:
        return [v.tolist() for v in vecs]
    else:
        return [list(map(float, v)) for v in vecs]


def _embed_with_openai(texts: List[str], model_name: str = OPENAI_MODEL_DEFAULT) -> List[List[float]]:
    if not _HAS_OPENAI:
        raise RuntimeError("openai package not installed or OPENAI_API_KEY missing")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    results = []
    # OpenAI supports batching; do simple loop to be robust
    for t in texts:
        resp = openai.Embedding.create(input=t, model=model_name)
        vec = resp["data"][0]["embedding"]
        results.append([float(x) for x in vec])
    return results


def _embed_with_hf_inference(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """
    Uses Hugging Face Inference API (embedding endpoint).
    Requires HF_INFERENCE_API env var (Bearer token).
    """
    if not HF_INFERENCE_API:
        raise RuntimeError("HF_INFERENCE_API env var not set for HuggingFace inference")
    headers = {"Authorization": f"Bearer {HF_INFERENCE_API}"}
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name or 'sentence-transformers/all-MiniLM-L6-v2'}"
    results = []
    for t in texts:
        r = requests.post(url, headers=headers, json={"inputs": t})
        r.raise_for_status()
        vec = r.json()
        # sometimes returns nested lists
        if isinstance(vec, list) and len(vec) and isinstance(vec[0], list):
            # collapse by averaging if multiple tokens
            import statistics
            dim = len(vec[0])
            avg = [statistics.mean(col) for col in zip(*vec)]
            results.append([float(x) for x in avg])
        else:
            results.append([float(x) for x in vec])
    return results


def embed_texts(texts: List[str], provider: Optional[str] = None, model: Optional[str] = None, batch_size: int = 16) -> List[List[float]]:
    """
    Embed a list of texts using the chosen provider. Returns list of vectors.
    """
    if provider is None:
        provider = DEFAULT_PROVIDER

    provider = provider.lower()

    if provider == "mock":
        return [_mock_vector_for_text(t) for t in texts]

    if provider == "sentence-transformers" or provider == "st":
        return _embed_with_sentence_transformers(texts, model or "all-MiniLM-L6-v2")

    if provider == "openai":
        return _embed_with_openai(texts, model or OPENAI_MODEL_DEFAULT)

    if provider in ("hf", "hf-inference", "huggingface"):
        return _embed_with_hf_inference(texts, model)

    # unknown provider: fallback to mock
    logger.warning("Unknown embed provider '%s' - falling back to mock", provider)
    return [_mock_vector_for_text(t) for t in texts]
```0