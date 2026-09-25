"""A small word-level bigram model used by the text-generation API."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable


class BigramModel:
    """Learn next-word probabilities from a collection of text samples."""

    def __init__(self, corpus: Iterable[str]) -> None:
        self.bigram_probabilities = self._train(corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[\w']+", text.lower())

    def _train(self, corpus: Iterable[str]) -> dict[str, dict[str, float]]:
        bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for text in corpus:
            words = self._tokenize(text)
            for current_word, next_word in zip(words, words[1:]):
                bigram_counts[current_word][next_word] += 1

        probabilities: dict[str, dict[str, float]] = {}
        for current_word, next_word_counts in bigram_counts.items():
            total = sum(next_word_counts.values())
            probabilities[current_word] = {
                next_word: count / total
                for next_word, count in next_word_counts.items()
            }
        return probabilities

    def generate_text(self, start_word: str, length: int = 20) -> str:
        """Generate up to ``length`` words, beginning with ``start_word``."""
        if length < 1:
            raise ValueError("length must be at least 1")

        current_word = start_word.strip().lower()
        if not current_word:
            raise ValueError("start_word must not be empty")

        generated_words = [current_word]
        for _ in range(length - 1):
            next_words = self.bigram_probabilities.get(current_word)
            if not next_words:
                break

            current_word = random.choices(
                list(next_words), weights=list(next_words.values()), k=1
            )[0]
            generated_words.append(current_word)

        return " ".join(generated_words)
