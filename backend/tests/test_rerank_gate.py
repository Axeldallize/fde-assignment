from backend.retrieval.rerank import rerank_by_heuristics
from backend.retrieval.gate import evidence_gate


def test_rerank_promotes_coverage():
    query = "introduction methods"
    candidates = [("a", 0.5), ("b", 0.5)]
    chunk_text_map = {
        "a": "This section covers the introduction and methods in detail.",
        "b": "Unrelated content without key terms.",
    }
    ranked = rerank_by_heuristics(query, candidates, chunk_text_map, headings_map=None, top_k=2)
    assert ranked[0][0] == "a"


def test_gate_pass_and_fail():
    # Pass: two distinct docs and decent sims
    ranked = [("a::ch1", 0.6), ("b::ch2", 0.5), ("a::ch3", 0.4)]
    chunk_doc_map = {"a::ch1": "a", "b::ch2": "b", "a::ch3": "a"}
    passed, meta = evidence_gate(ranked, chunk_doc_map, min_sources=2, threshold=0.3)
    assert passed is True
    assert meta["distinct_docs"] >= 2

    # Fail: low similarities and single doc
    ranked2 = [("a::ch1", 0.05), ("a::ch2", 0.04)]
    chunk_doc_map2 = {"a::ch1": "a", "a::ch2": "a"}
    passed2, meta2 = evidence_gate(ranked2, chunk_doc_map2, min_sources=2, threshold=0.5)
    assert passed2 is False
    assert meta2["distinct_docs"] == 1


