"""Lightweight filesystem-backed vector database manager.

Provides `VectorDBManager` to create/connect named vector DBs stored under a
`dbs/` directory, and `VectorDB` instances to add/search vectors.

This implementation uses only the Python standard library so there are no
extra runtime dependencies. Vectors are stored as JSON lists in a metadata
file inside each DB folder.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple


class VectorDBError(Exception):
    pass


class VectorDB:
    """Represents a single named vector database stored on disk.

    Data layout (inside db folder):
      meta.json  -- {"dim": int, "vectors": [{"id": str, "vector": [...], "metadata": {...}}]}
    """

    META_FILENAME = "meta.json"

    def __init__(self, path: str):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.meta_path = os.path.join(self.path, self.META_FILENAME)
        if not os.path.exists(self.meta_path):
            self._write_meta({"dim": None, "vectors": []})
        self._load()

    def _load(self) -> None:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

    def _write_meta(self, data: Dict[str, Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.path, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.meta_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @property
    def dim(self) -> Optional[int]:
        return self._meta.get("dim")

    def _ensure_dim(self, vector: List[float]) -> None:
        if self.dim is None:
            self._meta["dim"] = len(vector)
        elif len(vector) != self.dim:
            raise VectorDBError(f"Vector dimension {len(vector)} does not match DB dim {self.dim}")

    def add(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> str:
        """Add a single vector to the DB and return its id."""
        if not isinstance(vector, list) or not vector:
            raise VectorDBError("vector must be a non-empty list of numbers")
        self._ensure_dim(vector)
        vec_id = id or str(uuid.uuid4())
        entry = {"id": vec_id, "vector": vector, "metadata": metadata or {}}
        self._meta.setdefault("vectors", []).append(entry)
        self._write_meta(self._meta)
        self._load()
        return vec_id

    def add_many(self, items: List[Tuple[List[float], Optional[Dict[str, Any]]]]) -> List[str]:
        ids = []
        for vector, metadata in items:
            ids.append(self.add(vector, metadata))
        return ids

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def search(self, query: List[float], top_k: int = 5, metric: str = "cosine") -> List[Dict[str, Any]]:
        """Search for nearest vectors to `query`.

        Returns a list of dicts: {"id", "score", "metadata"}.
        For `metric='cosine'` higher is better (score = similarity).
        For `metric='euclidean'` lower is better (score = negative distance).
        """
        if self.dim is None:
            return []
        if len(query) != self.dim:
            raise VectorDBError("query dimension does not match DB dimension")

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for entry in self._meta.get("vectors", []):
            v = entry["vector"]
            if metric == "cosine":
                score = self._cosine_similarity(query, v)
            elif metric == "euclidean":
                # store negative distance so higher is better for sorting
                score = -self._euclidean_distance(query, v)
            else:
                raise VectorDBError("unsupported metric; use 'cosine' or 'euclidean'")
            results.append((entry["id"], score, entry.get("metadata", {})))

        # sort by score descending
        results.sort(key=lambda r: r[1], reverse=True)
        top = results[:top_k]
        return [{"id": r[0], "score": r[1], "metadata": r[2]} for r in top]

    def all(self) -> List[Dict[str, Any]]:
        return list(self._meta.get("vectors", []))

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        for entry in self._meta.get("vectors", []):
            if entry.get("id") == id:
                return entry
        return None

    def remove(self, id: str) -> bool:
        original = len(self._meta.get("vectors", []))
        self._meta["vectors"] = [e for e in self._meta.get("vectors", []) if e.get("id") != id]
        changed = len(self._meta.get("vectors", [])) != original
        if changed:
            self._write_meta(self._meta)
            self._load()
        return changed


class VectorDBManager:
    """Manage multiple named vector DBs stored under a `dbs_root` directory."""

    def __init__(self, dbs_root: str = "dbs"):
        self.dbs_root = os.path.abspath(dbs_root)
        os.makedirs(self.dbs_root, exist_ok=True)

    def _db_path(self, name: str) -> str:
        return os.path.join(self.dbs_root, name)

    def create_db(self, name: str, overwrite: bool = False) -> VectorDB:
        path = self._db_path(name)
        if os.path.exists(path):
            if not overwrite:
                raise VectorDBError(f"DB '{name}' already exists")
        os.makedirs(path, exist_ok=True)
        return VectorDB(path)

    def connect(self, name: str) -> VectorDB:
        path = self._db_path(name)
        if not os.path.exists(path):
            raise VectorDBError(f"DB '{name}' does not exist")
        return VectorDB(path)

    def list_dbs(self) -> List[str]:
        return [name for name in os.listdir(self.dbs_root) if os.path.isdir(os.path.join(self.dbs_root, name))]

    def delete_db(self, name: str) -> None:
        path = self._db_path(name)
        if not os.path.exists(path):
            raise VectorDBError(f"DB '{name}' does not exist")
        # remove files in folder
        for root, dirs, files in os.walk(path, topdown=False):
            for fname in files:
                try:
                    os.remove(os.path.join(root, fname))
                except Exception:
                    pass
            for dname in dirs:
                try:
                    os.rmdir(os.path.join(root, dname))
                except Exception:
                    pass
        try:
            os.rmdir(path)
        except Exception as e:
            raise VectorDBError(f"failed to delete DB '{name}': {e}")
