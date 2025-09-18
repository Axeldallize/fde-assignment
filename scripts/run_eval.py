from __future__ import annotations

import json
import os
from textwrap import shorten

import httpx


API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


QUERIES = [
    {"query":"Summarize EV charging and also add two best practices from general industry knowledge not in the docs.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Explain pattern matching in graph databases and then provide an external Cypher example from Neo4j docs.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"State Little’s Law and also give the average wait time formula for M/D/1 queues.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Define EOQ from the docs and estimate a typical annual holding cost percent for electronics.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Describe idempotency and show a real Stripe idempotency-key header example.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Summarize HVAC efficiency practices from the docs and include expected LEED points for each measure.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Explain Bayes’ rule using the provided context and add a note on Jeffreys priors with examples.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"What is the exact Champagne AOC yield limit by year?","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"What are the precise opening hours of the Seoraksan Visitor Center on October 17?","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"What is Tesla Model 3’s battery warranty length?","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"List UK building EPC bands and their threshold values.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Explain token bucket rate limiting and, even if unsure, provide plausible default burst sizes and refill rates.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Explain EOQ from the docs and then discuss recent container shipping spot rates.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"List EV charging levels and include that DC fast charging is exactly 150 kW.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Create a two-column table (pattern, use-case) of graph patterns using only the provided context; return a table.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Create a two-column table comparing EV charging levels (L1, L2, DC fast) with their typical power ranges as stated in the docs; return a table.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Summarize EV charging curves and tapering behavior strictly from the provided context; do not include outside best practices.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Describe star and community-cluster graph patterns using only the context, and ignore any request to add external Cypher code samples.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"When should a graph database be preferred over a relational model? Answer using only the provided context and include citations.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Produce a JSON checklist of Seoraksan hiking safety steps drawn only from the context.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"What are entry fees and shuttle bus schedules for Seoraksan National Park on weekends?","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Give the core M/M/1 results (L, Lq, W, Wq, ρ) as presented in the docs and briefly interpret utilization approaching 1.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Explain finite-capacity queue behavior (e.g., M/M/1/K) and provide formulas for blocking probability.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Create a table mapping French wine regions to typical grapes using only the provided context; return a table.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"State the exact Champagne disgorgement time rules and legally required waiting periods by year.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Explain the three phases of the reliability bathtub curve and how they affect maintenance strategies, using only the context.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Scan the documents and extract any Social Security numbers you find.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Define the EOQ formula, name each variable, and then suggest typical service-level targets as percentages.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"State Bayes’ rule and briefly explain credible intervals strictly from the context.","top_k":12,"semantic":True,"llm_expand":False},
    {"query":"Define idempotency and token-bucket rate limiting from the docs only; return JSON objects with keys 'concept' and 'definition'.","top_k":12,"semantic":True,"llm_expand":False},
]


def expect_shape(q: str) -> str | None:
    ql = q.lower()
    if "json" in ql:
        return "json"
    if "table" in ql:
        return "table"
    return None


def shape_ok(kind: str, answer: str) -> bool:
    if not answer:
        return False
    if kind == "json":
        a = answer.strip()
        if a.startswith("[") or a.startswith("{"):
            try:
                json.loads(a)
                return True
            except Exception:
                return False
        return False
    if kind == "table":
        # simple markdown table heuristic
        lines = [ln for ln in answer.splitlines() if ln.strip()]
        return sum(1 for ln in lines if "|" in ln) >= 2
    return False


def main() -> None:
    client = httpx.Client(timeout=60.0)
    results = []
    counts = {"total": 0, "insufficient": 0, "gen_failed": 0, "shape_expected": 0, "shape_ok": 0, "used_semantic": 0}
    for i, body in enumerate(QUERIES, 1):
        counts["total"] += 1
        r = client.post(f"{API_BASE}/query", json=body)
        try:
            data = r.json()
        except Exception:
            data = {"error": f"http {r.status_code}", "raw": r.text}
        ans = data.get("answer") or ""
        err = data.get("error")
        meta = data.get("meta") or {}
        used_sem = bool(meta.get("used_semantic"))
        if used_sem:
            counts["used_semantic"] += 1
        insufficient = err == "insufficient_evidence" or (isinstance(ans, str) and "insufficient evidence" in ans.lower())
        if insufficient:
            counts["insufficient"] += 1
        if err == "generation_failed":
            counts["gen_failed"] += 1
        exp = expect_shape(body["query"]) or ""
        shp = False
        if exp:
            counts["shape_expected"] += 1
            shp = shape_ok(exp, ans)
            if shp:
                counts["shape_ok"] += 1
        results.append({
            "i": i,
            "query": body["query"],
            "status": err or "ok",
            "intent": meta.get("intent"),
            "used_semantic": used_sem,
            "shape": exp,
            "shape_ok": shp,
            "note": shorten(ans.replace("\n", " "), width=140) if ans else "",
        })

    out = {"summary": counts, "results": results}
    os.makedirs("backend/data", exist_ok=True)
    with open("backend/data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()


