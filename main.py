from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Learning API", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "Welcome to Help Chat API app"}




if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)