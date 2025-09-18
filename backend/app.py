from fastapi import FastAPI, UploadFile, File
from .config import settings
from .utils.logging import configure_logging
from .models.io import IngestResponse, QueryRequest, QueryResponse, Citation
from .ingestion.service import ingest_files
from pathlib import Path

from .retrieval.intent import detect_intent
from .retrieval.rewrite import deterministic_rewrite
from .index.lexical import search as lexical_search
from .index.semantic import semantic_search
from .index.fusion import weighted_sum, rrf
from .retrieval.rerank import rerank_by_heuristics
from .retrieval.gate import evidence_gate
from .index.chunkio import get_text_map_for_ids, get_meta_map_for_ids
from .generation.prompt import build_prompt
from .generation.llm import generate_answer
from .generation.evidence_check import evidence_filter


configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, debug=settings.debug)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.env,
        "semantic": settings.use_semantic,
        "rrf": settings.use_rrf,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    paths: list[Path] = []
    for f in files:
        # Save uploaded file to a temp path then pass to orchestrator which copies to docs dir
        tmp_path = Path("/tmp") / f.filename
        content = await f.read()
        tmp_path.write_bytes(content)
        paths.append(tmp_path)
    counts = ingest_files(paths)
    return IngestResponse(ingested=[p.stem for p in paths], chunks=counts["chunks"], warnings=[])


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # Intent detection
    intent_res = detect_intent(req.query)
    if intent_res.intent == "smalltalk":
        msg = build_prompt("smalltalk", req.query, [])
        return QueryResponse(answer=msg, citations=[], meta={"intent": "smalltalk", "threshold_passed": False})

    # Rewrite
    q = deterministic_rewrite(req.query)

    # Retrieval
    # Guard lexical/semantic with best-effort fallbacks
    try:
        lex = lexical_search(q, top_k=req.top_k)
    except Exception:
        lex = []
    try:
        sem = semantic_search(q, top_k=req.top_k) if settings.use_semantic and req.semantic else []
    except Exception:
        sem = []
    fused = (
        rrf(lex, sem, top_k=req.top_k) if settings.use_rrf else weighted_sum(lex, sem, top_k=req.top_k)
    )

    # Build maps for rerank and citations
    chunk_ids = [cid for cid, _ in fused]
    id2text = get_text_map_for_ids(chunk_ids)
    id2meta = get_meta_map_for_ids(chunk_ids)
    id2heading = {cid: "/".join(id2meta.get(cid, {}).get("headings_path", []) or []) for cid in chunk_ids}
    id2doc = {cid: id2meta.get(cid, {}).get("doc_id", "?") for cid in chunk_ids}

    reranked = rerank_by_heuristics(req.query, fused, id2text, id2heading, top_k=req.top_k)

    # Gate
    passed, gate_meta = evidence_gate(reranked, id2doc)
    if not passed:
        return QueryResponse(error="insufficient_evidence", reason="gate_failed", citations=[], meta=gate_meta)

    # Assemble context (top-k small) and prompt
    top_ids = [cid for cid, _ in reranked[: min(4, len(reranked))]]
    context_texts = [id2text[cid] for cid in top_ids if cid in id2text]
    prompt = build_prompt("qa" if req.mode in ("auto", "qa") else req.mode, req.query, context_texts)

    # Generate
    try:
        answer = generate_answer(prompt, temperature=0.1)
    except Exception:
        # If LLM fails, return insufficient evidence rather than 500
        return QueryResponse(error="generation_failed", reason="llm_error", citations=[], meta={"intent": intent_res.intent})
    # Evidence filter
    answer_filtered = evidence_filter(answer, context_texts)

    # Citations
    citations: list[Citation] = []
    for cid, score in reranked[: min(4, len(reranked))]:
        meta = id2meta.get(cid, {})
        citations.append(
            Citation(
                doc_id=str(meta.get("doc_id", "?")),
                pages=f"{meta.get('page_start', '?')}-{meta.get('page_end', '?')}",
                heading=("/".join(meta.get("headings_path", []) or []) or None),
                score=float(score),
            )
        )

    return QueryResponse(
        answer=answer_filtered,
        citations=citations,
        meta={"intent": intent_res.intent, "threshold_passed": True, "used_semantic": bool(sem)},
    )

