from backend.retrieval.intent import detect_intent
from backend.retrieval.rewrite import deterministic_rewrite


def test_intent_smalltalk():
    r = detect_intent("Hello there")
    assert r.intent == "smalltalk" and r.confidence >= 0.8


def test_intent_qa():
    r = detect_intent("What is the method?")
    assert r.intent == "qa" and r.confidence >= 0.8


def test_rewrite_normalization():
    q = "   This   is\nA   Test  "
    out = deterministic_rewrite(q)
    assert out == "this is a test"


