# learning/encoding/feature_encoder.py
"""
Feature extractor for chunks.

Produces a lightweight feature vector/dict for each chunk including:
- token_count
- sentence_count
- avg_sentence_length
- readability_estimate (Flesch reading ease approximation if possible)
- uppercase_ratio
- digit_ratio
- named_entity_count (best-effort)
"""
from typing import Dict, Any
import re
import math
import logging

logger = logging.getLogger(__name__)

try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False


def _sentence_split(text: str) -> list:
    # naive sentence splitter
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _flesch_reading_ease(total_words: int, total_sentences: int, total_syllables: int) -> float:
    if total_sentences == 0 or total_words == 0:
        return 0.0
    # Flesch Reading Ease = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    return 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)


def _estimate_syllables(word: str) -> int:
    # very rough heuristic syllable estimation
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    # adjust
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)


def extract_features(text: str, tokenizer=None) -> Dict[str, Any]:
    """
    Return a dict of features for the given text.
    """
    if not text:
        return {}

    # token count (words)
    tokens = text.split()
    total_words = len(tokens)
    sentences = _sentence_split(text)
    total_sentences = len(sentences)

    # syllable estimate
    total_syllables = sum(_estimate_syllables(w) for w in tokens)

    avg_sentence_len = (total_words / total_sentences) if total_sentences else total_words

    uppercase_chars = sum(1 for c in text if c.isupper())
    digit_chars = sum(1 for c in text if c.isdigit())

    uppercase_ratio = uppercase_chars / max(1, len(text))
    digit_ratio = digit_chars / max(1, len(text))

    # best-effort named entity count via spaCy if available
    named_entities = 0
    entities_sample = []
    if _HAS_SPACY:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:10000])  # limit for speed
            named_entities = len(doc.ents)
            entities_sample = [ent.label_ for ent in doc.ents[:20]]
        except Exception:
            logger.exception("spaCy failed, continuing without NER")

    features = {
        "token_count": total_words,
        "sentence_count": total_sentences,
        "avg_sentence_length": avg_sentence_len,
        "estimated_syllables": total_syllables,
        "flesch_reading_ease": _flesch_reading_ease(total_words, total_sentences, total_syllables),
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "named_entity_count": named_entities,
        "named_entity_sample": entities_sample,
    }
    return features