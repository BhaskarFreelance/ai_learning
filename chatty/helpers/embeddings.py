from __future__ import annotations

import hashlib
import re
from typing import Sequence


class EmbeddingModel:
    """A lightweight embedding wrapper that avoids the blocked PyTorch/scikit stack."""

    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        self.name = name

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [token for token in text.split() if token]

    def encode(self, texts: Sequence[str] | str):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            tokens = self._tokenize(text)
            vector = []
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                value = int.from_bytes(digest[:8], "big") / 2**64
                vector.append(value)
            if not vector:
                vector = [0.0]
            embeddings.append(vector)
        return embeddings


_model_cache = {}


def get_embed_model(name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    if name not in _model_cache:
        _model_cache[name] = EmbeddingModel(name)
    return _model_cache[name]
