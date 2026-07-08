import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Chatty API", version="1.0.0")


@app.get("/")
def read_root():
    return {"message": "Welcome to Chatty!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
