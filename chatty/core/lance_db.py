"""Adapter for using Lancedb as the vector database backend.

This is a small wrapper that exposes a familiar `create_db`, `connect`, and
`add`/`search` surface. It requires `lancedb` to be installed. If `lancedb`
is missing the adapter will raise a helpful error at runtime.
"""
import json
import os
from typing import Any, Dict, List, Optional

try:
    import pyarrow as pa
except Exception:  # pragma: no cover - optional dependency
    pa = None

try:
    import lancedb
except Exception:  # ImportError or runtime errors
    lancedb = None


class LanceDBError(Exception):
    pass


class LanceCollection:
    def __init__(self, client: Any, name: str):
        self.client = client
        self.name = name
        self.table = None
        # try to open existing table/collection; creation handled at manager
        try:
            if hasattr(client, "open_table"):
                self.table = client.open_table(name)
            elif hasattr(client, "table"):
                self.table = client.table(name)
            else:
                self.table = client[name]
        except Exception:
            self.table = None

    def add(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> str:
        if lancedb is None:
            raise LanceDBError("lancedb is not installed; install it to use LanceDB backend")
        if self.table is None:
            raise LanceDBError("LanceDB table is not available")
        # Try several possible insert/upsert methods depending on lancedb version
        row = {"embedding": vector, "metadata": json.dumps(metadata or {})}
        if id is not None:
            row["id"] = id
        try:
            if hasattr(self.table, "insert"):
                self.table.insert([row])
            elif hasattr(self.table, "upsert"):
                self.table.upsert([row])
            elif hasattr(self.table, "add"):
                self.table.add([row])
            else:
                raise LanceDBError("unable to insert rows into Lance table with current client API")
        except Exception as e:
            raise LanceDBError(f"failed to add vector: {e}")
        return id or ""

    def search(self, query: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if lancedb is None:
            raise LanceDBError("lancedb is not installed; install it to use LanceDB backend")
        if self.table is None:
            raise LanceDBError("LanceDB table is not available")
        # Try the common search API shapes
        try:
            if hasattr(self.table, "search"):
                try:
                    res = self.table.search(query, vector_column_name="embedding")
                except TypeError:
                    res = self.table.search(query)

                if hasattr(res, "limit"):
                    res = res.limit(top_k)

                rows = []
                if hasattr(res, "to_list"):
                    try:
                        rows = res.to_list()
                    except Exception:
                        rows = []
                elif hasattr(res, "to_arrow"):
                    try:
                        rows = res.to_arrow().to_pylist()
                    except Exception:
                        rows = []
                else:
                    try:
                        rows = list(res)
                    except Exception:
                        rows = []

                out = []
                for hit in rows:
                    try:
                        score = hit.get("_distance", 0) if isinstance(hit, dict) else getattr(hit, "score", None) or getattr(hit, "distance", None) or 0
                    except Exception:
                        score = 0

                    metadata = None
                    if isinstance(hit, dict):
                        metadata = hit.get("metadata")
                    else:
                        metadata = getattr(hit, "metadata", None)

                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except Exception:
                            metadata = {}
                    elif metadata is None:
                        metadata = {}

                    out.append({
                        "id": None,
                        "score": score,
                        "metadata": metadata or {},
                    })
                return out
            else:
                raise LanceDBError("table.search not available on this lancedb client version")
        except Exception as e:
            raise LanceDBError(f"search failed: {e}")


class LanceDBManager:
    def __init__(self, dbs_root: str = "dbs"):
        if lancedb is None:
            raise LanceDBError("lancedb package not installed; please pip install lancedb")
        self.dbs_root = os.path.abspath(dbs_root)
        os.makedirs(self.dbs_root, exist_ok=True)
        # connect to root directory; lancedb client will manage named tables
        try:
            self.client = lancedb.connect(self.dbs_root)
        except Exception as e:
            # Some lancedb installs use Client class
            try:
                self.client = lancedb.Client(self.dbs_root)
            except Exception:
                raise LanceDBError(f"failed to connect lancedb at {self.dbs_root}: {e}")

    def create_db(self, name: str, overwrite: bool = False) -> LanceCollection:
        # create a table/collection with name
        try:
            if hasattr(self.client, "create_table"):
                if overwrite:
                    try:
                        if hasattr(self.client, "drop_table"):
                            self.client.drop_table(name)
                    except Exception:
                        pass

                schema = None
                if pa is not None:
                    schema = pa.schema([
                        pa.field("embedding", pa.list_(pa.float32(), list_size=3)),
                        pa.field("metadata", pa.string()),
                    ])

                try:
                    self.client.create_table(name, schema=schema, exist_ok=True)
                except TypeError:
                    try:
                        self.client.create_table(name, schema=schema)
                    except Exception:
                        try:
                            self.client.create_table(name)
                        except Exception:
                            pass
                except Exception:
                    pass
            return LanceCollection(self.client, name)
        except Exception as e:
            raise LanceDBError(f"failed to create DB '{name}': {e}")

    def connect(self, name: str) -> LanceCollection:
        # check existence heuristically
        try:
            collection = LanceCollection(self.client, name)
            if collection.table is None and hasattr(self.client, "table_names"):
                try:
                    names = set(self.client.table_names())
                    if name not in names:
                        raise LanceDBError(f"DB '{name}' does not exist")
                except Exception:
                    pass
            return collection
        except Exception as e:
            raise LanceDBError(f"failed to open DB '{name}': {e}")

    def list_dbs(self) -> List[str]:
        # delegate to client if possible
        if hasattr(self.client, "table_names"):
            try:
                return list(self.client.table_names())
            except Exception:
                pass
        if hasattr(self.client, "tables"):
            try:
                return list(self.client.tables())
            except Exception:
                pass
        # fallback: list directories under root
        return [name for name in os.listdir(self.dbs_root) if os.path.isdir(os.path.join(self.dbs_root, name))]

    def delete_db(self, name: str) -> None:
        path = os.path.join(self.dbs_root, name)
        if os.path.exists(path):
            # remove directory
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        else:
            # try client-level drop
            if hasattr(self.client, "drop_table"):
                try:
                    self.client.drop_table(name)
                    return
                except Exception as e:
                    raise LanceDBError(f"failed to delete DB '{name}': {e}")
            raise LanceDBError(f"DB '{name}' does not exist")
