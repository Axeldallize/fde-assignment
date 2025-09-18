from backend.index.fusion import weighted_sum, rrf


def test_weighted_sum_basic():
    lex = [("a", 0.9), ("b", 0.5)]
    sem = [("b", 0.9), ("c", 0.8)]
    fused = weighted_sum(lex, sem, w_lex=0.5, w_sem=0.5, top_k=3)
    # b should rank top combining both
    assert fused[0][0] == "b"


def test_rrf_basic():
    lex = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    sem = [("c", 0.95), ("b", 0.6), ("d", 0.2)]
    fused = rrf(lex, sem, k=60, top_k=4)
    ids = [cid for cid, _ in fused]
    assert set(ids) == {"a", "b", "c", "d"}


