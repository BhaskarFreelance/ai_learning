from fastapi import APIRouter, HTTPException, Body
from typing import Optional, List, Dict, Any
import os
import time
import json

try:
    from chatty.helpers.embeddings import EmbeddingModel, get_embed_model as get_embedding_model
except ImportError:
    from helpers.embeddings import EmbeddingModel, get_embed_model as get_embedding_model

try:
    from chatty.helpers.hf_auth import configure_hf_token
except ImportError:
    from helpers.hf_auth import configure_hf_token

configure_hf_token()

from .lance_db import LanceDBManager, LanceDBError
from .sqlite_manager import SQLiteManager, ChatSessionManager, APILogManager

try:
    from chatty.helpers.connection import OppenAI
except ImportError:
    from helpers.connection import OppenAI

router = APIRouter(prefix="/chat")

_embed_model_cache: Dict[str, EmbeddingModel] = {}


def get_embed_model(name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    if name not in _embed_model_cache:
        _embed_model_cache[name] = get_embedding_model(name)
    return _embed_model_cache[name]


@router.post("/")
def chat(
    db: str = Body(..., embed=True),
    message: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    top_k: int = Body(5, embed=True),
    embed_model: str = Body("all-MiniLM-L6-v2", embed=True),
    openai_model: str = Body("gpt-4o-mini", embed=True),
    temperature: float = Body(0.7, embed=True),
):
    """Retrieve relevant docs from the named vector DB, then call OpenAI chat completion.

    Request body (JSON): {"db": "name", "message": "...", "top_k": 5}
    """
    # locate DBs root (one level up from core)
    dbs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dbs"))
    try:
        manager = LanceDBManager(dbs_root=dbs_root)
        vectordb = manager.connect(db)
    except LanceDBError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # prepare managers: chats and logs
    chats_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dbs", "chats.db"))
    logs_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dbs", "logs.db"))
    chat_mgr = ChatSessionManager(chats_db)
    log_mgr = APILogManager(logs_db)

    # ensure or create session
    if session_id and chat_mgr.session_exists(session_id):
        sid = session_id
    else:
        sid = chat_mgr.create_session(db)

    # log start
    start = time.time()

    # record user's message in session
    chat_mgr.add_message(sid, "user", message)

    # embed the query
    try:
        embedder = get_embed_model(embed_model)
        qvec = embedder.encode([message])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to compute embedding: {e}")

    try:
        results = vectordb.search(list(map(float, qvec.tolist() if hasattr(qvec, "tolist") else qvec)), top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

    # build context from results
    context_texts = [r.get("metadata", {}).get("text", "") for r in results]
    context_combined = "\n\n---\n\n".join(context_texts)

    # load prompts
    prompts_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dbs", "prompts.db"))
    pmgr = SQLiteManager(prompts_db)
    prompts = pmgr.get_prompts(db) or {}
    system_prompt = prompts.get("system") or "You are a helpful assistant."
    assistant_prompt = prompts.get("assistant") or ""

    # assemble messages for OpenAI
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        # include retrieved context in system message to keep it visible
        messages.append({"role": "system", "content": system_prompt + "\n\nRelevant documents:\n" + context_combined})
    if assistant_prompt:
        messages.append({"role": "assistant", "content": assistant_prompt})
    # user's message
    user_content = f"{message}\n\nUse the above documents to answer the user concisely."
    messages.append({"role": "user", "content": user_content})

    # call OpenAI
    try:
        ai = OppenAI()
        resp = ai.chat(messages=messages, model=openai_model, temperature=temperature)
        # extract assistant text
        assistant_text = ""
        try:
            assistant_text = resp["choices"][0]["message"]["content"]
        except Exception:
            assistant_text = str(resp)
    except Exception as e:
        # log failure
        latency_ms = (time.time() - start) * 1000.0
        try:
            payload = json.dumps({"db": db, "message": message[:1000]})
        except Exception:
            payload = ""
        try:
            log_mgr.log("/chat", "POST", payload, 500, latency_ms)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}")

    # store assistant reply in session
    chat_mgr.add_message(sid, "assistant", assistant_text)

    # log success
    latency_ms = (time.time() - start) * 1000.0
    try:
        payload = json.dumps({"db": db, "message": message[:1000]})
    except Exception:
        payload = ""
    try:
        log_mgr.log("/chat", "POST", payload, 200, latency_ms)
    except Exception:
        pass

    history = chat_mgr.get_history(sid)
    return {"reply": assistant_text, "sources": results, "session_id": sid, "history": history}
