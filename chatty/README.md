Chatty — simple FastAPI app for vector DB backed QA + chat

Setup

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add your OpenAI API key to `chatty/.env`:

```
OPENAI_API_KEY=sk-...
```

Run

```powershell
uvicorn chatty.main:app --reload --port 8080
```

Endpoints

- POST `/vector_db/create` (form-data)
  - `name` (string, required)
  - `file` (file) or `text` (string) or `url` (string)
  - `chunk_size` (int, optional)
  - `overlap` (int, optional)
  - `model_name` (string, optional)

  Creates a vector DB stored under `chatty/dbs/<name>` and indexes content using `sentence-transformers` embeddings.

- POST `/prompts/set` (form)
  - `name` (string, required)
  - `system_prompt` (string, optional)
  - `assistant_prompt` (string, optional)

  Stores system/assistant prompts for a named DB (stored in `chatty/dbs/prompts.db`).

- GET `/prompts/{name}` — retrieve stored prompts for a DB.

- POST `/chat/` (JSON)
  - `db` (string, required)
  - `message` (string, required)
  - `session_id` (string, optional)
  - `top_k`, `embed_model`, `openai_model`, `temperature` (optional)

  Performs retrieval from the named vector DB, composes messages with stored prompts, calls OpenAI chat completion, stores chat history in `chatty/dbs/chats.db`, and logs the request to `chatty/dbs/logs.db`.

Notes

- The app uses simple filesystem-backed JSON vector stores (`chatty/core/vector_db.py`) — suitable for prototypes only.
- `sentence-transformers` will download models on first use; ensure network connectivity and enough disk space.
- Keep `chatty/.env` out of version control.

If you want, I can also:
- Add example `curl` requests for each endpoint
- Add automated tests or a lightweight CLI for common workflows
*** End Patch