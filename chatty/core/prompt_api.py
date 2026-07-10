from fastapi import APIRouter, Form, HTTPException
from typing import Optional
import os

from .sqlite_manager import SQLiteManager

router = APIRouter(prefix="/prompts")


def _prompts_db_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dbs", "prompts.db"))


@router.post("/set")
def set_prompts(
    name: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    assistant_prompt: Optional[str] = Form(None),
):
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    db_path = _prompts_db_path()
    mgr = SQLiteManager(db_path)
    try:
        mgr.set_prompts(name, system_prompt, assistant_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"db": name, "ok": True}


@router.get("/{name}")
def get_prompts(name: str):
    db_path = _prompts_db_path()
    mgr = SQLiteManager(db_path)
    data = mgr.get_prompts(name)
    if data is None:
        raise HTTPException(status_code=404, detail="prompts not found")
    return data


@router.delete("/{name}")
def delete_prompts(name: str):
    db_path = _prompts_db_path()
    mgr = SQLiteManager(db_path)
    ok = mgr.delete_prompts(name)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"db": name, "deleted": True}


@router.get("/")
def list_prompts():
    db_path = _prompts_db_path()
    mgr = SQLiteManager(db_path)
    return {"items": mgr.list_prompts()}
