# rag-agent-pipeline

Initial project scaffold for a RAG + agent workflow.

## Project structure

```text
rag-agent-pipeline/
├── app/
│   ├── api/      # FastAPI routes
│   ├── agent/    # LangGraph graphs
│   ├── rag/      # RAG pipeline logic
│   ├── llm/      # Ollama client
│   ├── db/       # ChromaDB + SQLite access
│   ├── config/   # settings.py
│   └── main.py
├── data/
│   ├── raw/      # source docs
│   └── chroma/   # vector DB files
├── tests/
├── scripts/
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Quick start

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest
```
