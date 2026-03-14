from fastapi import FastAPI

from app.api.routes import router as api_router


app = FastAPI(title="RAG Agent Pipeline")
app.include_router(api_router, prefix="/api")


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "rag-agent-pipeline"}
