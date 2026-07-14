import uvicorn
from fastapi import FastAPI

try:
    from chatty.core.vector_api import router as vector_router
    from chatty.core.prompt_api import router as prompt_router
    from chatty.core.chat_api import router as chat_router
except ImportError:
    from core.vector_api import router as vector_router
    from core.prompt_api import router as prompt_router
    from core.chat_api import router as chat_router

app = FastAPI(title="Chatty API", version="1.0.0")


@app.get("/")
def read_root():
    return {"message": "Welcome to Chatty!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(vector_router)
app.include_router(prompt_router)
app.include_router(chat_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
