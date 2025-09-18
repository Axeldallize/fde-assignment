# RAG Pipeline (FastAPI + React) — Assignment

This repo contains a minimal Retrieval-Augmented Generation (RAG) backend and UI scaffold. Priority: accuracy > simplicity > speed. Phase 0 includes skeleton, config, JSON logging, and a health endpoint.

## Quickstart

1) Python env

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Environment

Create a `.env` file in the project root based on the keys below:

```
ENV=dev
LOG_LEVEL=INFO
DEBUG=false
USE_SEMANTIC=true
USE_RRF=false
EVIDENCE_TOPK=4
EVIDENCE_THRESHOLD=0.28
LLM_PROVIDER=mistral
MISTRAL_MODEL=mistral-small-2506 # Note: I use the smaller model to mminimize costs for StackAI, which also allows for lower latency, but less accurate results as a trade-off
EMBEDDING_PROVIDER=mistral
EMBEDDING_MODEL=mistral-embed
# MISTRAL_API_KEY=...
```

3) Run API

```
uvicorn backend.app:app --reload
```

Visit `/health` to check readiness.

4) UI (placeholder)

```
cd ui
npm i
npm run dev
```

## Decisions and considerations
- Python 3.12, pip+venv
- No auth/rate limit (private demo)
- Fusion default: normalized weighted-sum (RRF flag available, default false)
- Embeddings default: Mistral API
- Persistence: transient files under `backend/data/` (re-ingest acceptable)

## Next phases
- Phase 1: ingestion 
- Phase 2: query 
- Phase 3: minimal UI
- Phase 4: tests
