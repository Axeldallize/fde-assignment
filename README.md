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
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-4-20250514
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-3.5
# ANTHROPIC_API_KEY=...
# VOYAGE_API_KEY=...
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
- Embeddings default: Voyage API
- Persistence: transient files under `backend/data/` (re-ingest acceptable)

## Next phases
- Phase 1: ingestion — DONE
  - PDF extraction (PyMuPDF): per-page text + heading candidates
  - Hybrid chunking: heading-bounded with ~15% overlap (target ~1k tokens)
  - Persistence: JSONL chunks + texts/map sidecars
  - Lexical index: TF‑IDF (1–2 grams, english stopwords), saved vectorizer/matrix/ids
  - Orchestrator + endpoint: `/ingest` wires extract→chunk→persist→manifest→rebuild index (and embeddings if semantic enabled)
  - Smoke tests: `scripts/pdf_extract_smoke.py`, `scripts/chunk_smoke.py`, `scripts/build_tfidf_and_query_smoke.py`
- Phase 2: query — DONE
  - Intent detection (smalltalk/qa) and deterministic rewrite
  - Retrieval: lexical (TF‑IDF) + semantic (Voyage); fusion (weighted‑sum default; RRF optional)
  - Rerank: coverage + heading bonus
  - Evidence gate: mean top‑k similarity + multi‑source requirement
  - Generation: Anthropic (Claude) with templates (qa/list/table); smalltalk politely refuses
  - Evidence check: sentence‑level support filter
- Phase 3: minimal UI
- Phase 4: tests

### Ingestion quickstart

1) Start API and open docs
```
source .venv/bin/activate
python -m uvicorn backend.app:app --reload --port 8000
open http://localhost:8000/docs
```

2) Use POST `/ingest` to upload one or more PDFs (field name `files`).

3) Verify artifacts
```
ls -lah backend/data/docs/
cat backend/data/manifests/manifest.json | sed -n '1,120p'
ls -lah backend/data/chunks/
head -n 2 backend/data/chunks/<doc_id>.jsonl
cat backend/data/chunks/<doc_id>.texts.json | sed -n '1,80p'
cat backend/data/chunks/<doc_id>.map.json | sed -n '1,80p'
ls -lah backend/data/index/
```

4) Optional: lexical retrieval smoke test
```
PYTHONPATH=$PWD python scripts/build_tfidf_and_query_smoke.py
```

### Query quickstart

1) Ensure embeddings are built if using semantic
```
PYTHONPATH=$PWD python - <<'PY'
from backend.index.semantic import build_embeddings_from_all_chunks
build_embeddings_from_all_chunks()
print("Embeddings built.")
PY
```

2) Query via Swagger (POST `/query`) or curl
```
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"what is covered in the introduction?","mode":"auto","top_k":12,"semantic":true,"llm_expand":false}'
```

Response includes `answer`, `citations`, and `meta` (intent, threshold_passed, used_semantic).

## System architecture

```mermaid
flowchart LR
  UI[React/Vite UI] -->|/ingest| API[FastAPI]
  UI -->|/query| API

  subgraph Backend
    API --> INTENT[Intent detector]
    INTENT -->|smalltalk| RESP[Polite refusal]
    INTENT -->|qa| REW[Query rewrite]

    REW --> LEX[TF‑IDF lexical index]
    REW --> SEM[Embeddings index (Voyage)]
    LEX --> FUSE[Fusion]
    SEM --> FUSE
    FUSE --> RER[Heuristic reranker]
    RER --> GATE[Evidence gate]
    GATE -->|fail| IE[Insufficient evidence]
    GATE -->|pass| PROMPT[Prompt builder]
    PROMPT --> LLM[Anthropic Claude]
    LLM --> EC[Sentence‑level evidence filter]
  end

  EC --> RESP
  IE --> RESP
```

## Design decisions and trade‑offs (highlights)

- Chunking: hybrid, heading‑bounded (~1k tokens target) with ~15% overlap. Simpler than full layout parsing but preserves topical boundaries and improves recall.
- Headings: heuristic detection (numbered headings, ALL‑CAPS short lines, title‑case). Lightweight and robust for mixed PDFs.
- Lexical retrieval: TF‑IDF (1–2 grams, english stopwords), L2‑normalized; fast, explainable backstop for exact terms.
- Semantic retrieval: Voyage `voyage-3.5` embeddings, cosine similarity, flat NumPy search (no external vector DB) to stay framework‑free.
- Fusion: normalized weighted‑sum (default) with optional RRF flag for rank‑robust fusion across query styles.
- Reranker: heuristic boost for query‑term coverage and heading match; favors diverse, better‑supported chunks.
- Gate: requires mean top‑k similarity ≥ threshold (default 0.28) and multiple distinct sources; reduces hallucinations by refusing weak evidence.
- Generation: Anthropic `claude-sonnet-4-20250514`, low temperature (0.1); prompt templates for qa/list/table; smalltalk politely refused.
- Evidence filter: sentence‑level cosine vs. context; drops unsupported lines (threshold default 0.15) instead of fabricating.
- Safety: smalltalk refusal; no PII extraction unless explicitly found in corpus (gate+filter enforce).
- Persistence: file‑backed artifacts under `backend/data/`; rebuild lexical on ingest; rebuild embeddings when semantic enabled.
- Config & toggles: runtime overrides on `/query` (use_rrf, top_k, evidence_topk/threshold, temperature); UI exposes controls.

## Evaluation (probe set)

Run:
```
API_BASE=http://localhost:8000 python scripts/run_eval.py
```

Summary (from `backend/data/eval_results.json`):

```
{
  "total": 30,
  "insufficient": 24,
  "gen_failed": 0,
  "shape_expected": 5,
  "shape_ok": 2,
  "used_semantic": 29
}
```

Notes:
- High “insufficient” reflects strict evidence gating and filtering (by design). Numeric drift/adversarial prompts are curtailed.
- Shape checks passed for 2/5 where the corpus contained enough structure to render tables; others were refused or lacked structure.
- Semantic used in 29/30 indicating embeddings built and active.
- Full trace with per‑query notes lives at `backend/data/eval_results.json`.

## Libraries and software

- Backend
  - FastAPI (`https://fastapi.tiangolo.com/`)
  - Uvicorn (`https://www.uvicorn.org/`)
  - Pydantic (`https://docs.pydantic.dev/`)
  - PyMuPDF (`https://pymupdf.readthedocs.io/`)
  - NumPy (`https://numpy.org/`)
  - scikit‑learn (`https://scikit-learn.org/`)
  - httpx (`https://www.python-httpx.org/`)
  - Anthropic SDK (`https://docs.anthropic.com/claude/docs`)
  - Voyage AI (`https://docs.voyageai.com/`)

- Frontend
  - React (`https://react.dev/`)
  - Vite (`https://vitejs.dev/`)
  - react‑markdown (`https://github.com/remarkjs/react-markdown`)
  - remark‑gfm (`https://github.com/remarkjs/remark-gfm`)

