from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import os
import requests

from sentence_transformers import SentenceTransformer

from .lance_db import LanceDBManager, LanceDBError

router = APIRouter()

_model_cache = {}


def get_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - overlap
    return chunks


@router.post("/vector_db/create")
async def create_vector_db(
    name: str = Form(...),
    url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    chunk_size: int = Form(500),
    overlap: int = Form(50),
    model_name: str = Form("all-MiniLM-L6-v2"),
):
    """Create a named vector DB from uploaded file, raw text, or remote URL.

    - `name`: name of DB to create
    - provide one of `file`, `text`, or `url`
    - `chunk_size` and `overlap` control splitting
    - `model_name` is a sentence-transformers model id
    """
    if not (file or text or url):
        raise HTTPException(status_code=400, detail="Provide one of 'file', 'text', or 'url'")

    # obtain content
    content = ""
    if url:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            content = resp.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to fetch url: {e}")
    elif file:
        raw = await file.read()
        try:
            content = raw.decode("utf-8")
        except Exception:
            try:
                content = raw.decode("latin-1")
            except Exception:
                raise HTTPException(status_code=400, detail="could not decode uploaded file")
    else:
        content = text or ""

    if not content.strip():
        raise HTTPException(status_code=400, detail="Empty content")

    chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
    model = get_model(model_name)
    embeddings = model.encode(chunks)

    # prepare DB manager rooted in chatty/dbs (one level up from core)
    dbs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dbs"))
    try:
        manager = LanceDBManager(dbs_root=dbs_root)
        db = manager.create_db(name, overwrite=False)
    except LanceDBError as e:
        raise HTTPException(status_code=400, detail=str(e))

    added = 0
    for chunk, emb in zip(chunks, embeddings):
        try:
            vec = emb.tolist() if hasattr(emb, "tolist") else emb
            db.add(vector=list(map(float, vec)), metadata={"text": chunk})
            added += 1
        except Exception:
            continue

    return {"db": name, "created": True, "items_added": added}
