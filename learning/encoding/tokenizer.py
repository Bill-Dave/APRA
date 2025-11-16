# learning/encoding/tokenizer.py
"""
Tokenization utilities with graceful fallbacks.

Provides:
- get_tokenizer(model_name=None): returns an object with `.encode(text)` and `.decode(token_ids)` methods
- count_tokens(text, model_name=None): integer token count

Preferred tokenizer: tiktoken (OpenAI). Falls back to a word-based tokenizer.
"""
from typing import List, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


class _SimpleTokenizer:
    """Whitespace-and-punctuation tokenizer fallback."""

    def encode(self, text: str) -> List[str]:
        # naive tokenization: split on whitespace; keep punctuation attached
        return [tok for tok in text.replace("\n", " ").split() if tok]

    def decode(self, tokens: List[str]) -> str:
        return " ".join(tokens)


class _TiktokenAdapter:
    def __init__(self, model_name: Optional[str] = None):
        # default to cl100k_base (used by many OpenAI models)
        self.encoding = tiktoken.encoding_for_model(model_name) if model_name else tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)


def get_tokenizer(model_name: Optional[str] = None) -> Any:
    """
    Return a tokenizer object with `encode(text)` and `decode(tokens)`.

    Prefer tiktoken when available, else return the simple whitespace tokenizer.
    """
    if _HAS_TIKTOKEN:
        try:
            return _TiktokenAdapter(model_name=model_name)
        except Exception:
            logger.exception("tiktoken adapter failed, falling back")
    return _SimpleTokenizer()


def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    tok = get_tokenizer(model_name)
    try:
        encoded = tok.encode(text)
        return len(encoded)
    except Exception:
        # fallback: approximate by whitespace split
        return max(1, len(text.split()))