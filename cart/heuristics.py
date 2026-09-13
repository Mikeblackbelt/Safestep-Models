"""
Heuristic feature extraction for CART.

These are hand-engineered signals that complement the neural backbone
and improve robustness against adversarial paraphrasing: perplexity/
burstiness, n-gram repetition, and stylistic markers (punctuation,
sentence-length variance, capitalization consistency).

No external language model dependency is required: perplexity here is
approximated with a simple n-gram frequency model fit at extraction
time, which is enough to separate the two classes' statistics without
needing to ship a KenLM binary. Swap in a proper KenLM-backed
implementation later without changing the public interface.
"""

import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np

from cart.config import CARTConfig

FEATURE_NAMES = [
    "perplexity_proxy",
    "burstiness",
    "repetition_ratio",
    "unique_ngram_ratio",
    "max_ngram_repeat",
    "punctuation_regularity",
    "sentence_length_mean",
    "sentence_length_std",
    "capitalization_consistency",
    "rare_word_ratio",
    "transition_word_density",
    "avg_word_length",
]


class HeuristicExtractor:
    """Computes hand-engineered stylistic/statistical features from raw text."""

    _TRANSITION_WORDS = {
        "however", "therefore", "furthermore", "moreover", "additionally",
        "consequently", "nevertheless", "thus", "hence", "meanwhile",
        "in conclusion", "in summary", "overall", "importantly",
    }

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    _WORD_RE = re.compile(r"[A-Za-z']+")

    def __init__(self, config: CARTConfig):
        self.config = config

    def extract(self, text: str) -> Dict[str, float]:
        """Compute the full feature dict for a single text sample."""
        words = self._WORD_RE.findall(text)
        sentences = [s for s in self._SENTENCE_SPLIT_RE.split(text.strip()) if s]

        if not words:
            return {name: 0.0 for name in FEATURE_NAMES}

        features = {}
        features["perplexity_proxy"] = self._perplexity_proxy(words)
        features["burstiness"] = self._burstiness(words)
        rep_ratio, unique_ratio, max_repeat = self._repetition_features(words)
        features["repetition_ratio"] = rep_ratio
        features["unique_ngram_ratio"] = unique_ratio
        features["max_ngram_repeat"] = max_repeat
        features["punctuation_regularity"] = self._punctuation_regularity(text)
        sent_mean, sent_std = self._sentence_length_stats(sentences)
        features["sentence_length_mean"] = sent_mean
        features["sentence_length_std"] = sent_std
        features["capitalization_consistency"] = self._capitalization_consistency(sentences)
        features["rare_word_ratio"] = self._rare_word_ratio(words)
        features["transition_word_density"] = self._transition_word_density(text)
        features["avg_word_length"] = float(np.mean([len(w) for w in words]))

        return features

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Vectorize a batch of texts into a (N, num_features) array, ordered by FEATURE_NAMES."""
        rows = [self.extract(t) for t in texts]
        return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)

    # -- individual feature groups -------------------------------------

    def _perplexity_proxy(self, words: List[str]) -> float:
        """
        Approximate perplexity via unigram self-entropy: lower values
        indicate more predictable (repetitive/formulaic) text, which
        correlates with AI-generated output.
        """
        counts = Counter(w.lower() for w in words)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        # Normalize to a perplexity-like scale
        return float(2 ** entropy)

    def _burstiness(self, words: List[str]) -> float:
        """
        Variance in token frequency relative to mean; AI text tends to
        show lower burstiness (more uniform word usage) than human text.
        """
        counts = np.array(list(Counter(w.lower() for w in words).values()), dtype=np.float64)
        if len(counts) < 2 or counts.mean() == 0:
            return 0.0
        return float(counts.std() / counts.mean())

    def _repetition_features(self, words: List[str]):
        ngram_min, ngram_max = self.config.ngram_min, self.config.ngram_max
        all_ngrams = []
        for n in range(ngram_min, ngram_max + 1):
            all_ngrams.extend(
                tuple(w.lower() for w in words[i:i + n]) for i in range(len(words) - n + 1)
            )

        if not all_ngrams:
            return 0.0, 1.0, 0

        counts = Counter(all_ngrams)
        total = len(all_ngrams)
        unique = len(counts)
        repeated = sum(1 for c in counts.values() if c > self.config.repetition_threshold)

        repetition_ratio = repeated / unique if unique else 0.0
        unique_ngram_ratio = unique / total if total else 0.0
        max_repeat = max(counts.values())

        return float(repetition_ratio), float(unique_ngram_ratio), float(max_repeat)

    def _punctuation_regularity(self, text: str) -> float:
        """Variance in spacing before punctuation; low variance -> more regular/formal (AI-leaning)."""
        gaps = [m.start() for m in re.finditer(r"[.,;:!?]", text)]
        if len(gaps) < 2:
            return 0.0
        diffs = np.diff(gaps)
        if diffs.mean() == 0:
            return 0.0
        return float(diffs.std() / diffs.mean())

    def _sentence_length_stats(self, sentences: List[str]):
        if not sentences:
            return 0.0, 0.0
        lengths = [len(self._WORD_RE.findall(s)) for s in sentences]
        return float(np.mean(lengths)), float(np.std(lengths))

    def _capitalization_consistency(self, sentences: List[str]) -> float:
        """Fraction of sentences that start with a capital letter (AI text tends toward 1.0)."""
        if not sentences:
            return 0.0
        starts_upper = sum(1 for s in sentences if s and s[0].isupper())
        return starts_upper / len(sentences)

    def _rare_word_ratio(self, words: List[str]) -> float:
        """
        Proxy for lexical richness: fraction of words that appear only
        once in the sample. Human writing tends to have a higher ratio
        of singleton ("rare-in-context") words.
        """
        counts = Counter(w.lower() for w in words)
        singletons = sum(1 for c in counts.values() if c == 1)
        return singletons / len(counts) if counts else 0.0

    def _transition_word_density(self, text: str) -> float:
        text_lower = text.lower()
        word_count = len(self._WORD_RE.findall(text))
        if word_count == 0:
            return 0.0
        hits = sum(text_lower.count(tw) for tw in self._TRANSITION_WORDS)
        return hits / word_count