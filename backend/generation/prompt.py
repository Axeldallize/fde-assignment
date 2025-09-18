from __future__ import annotations

from typing import List, Literal, Dict


Mode = Literal["qa", "list", "table", "smalltalk"]


def build_prompt(mode: Mode, query: str, context_chunks: List[str]) -> str:
    if mode == "smalltalk":
        # For smalltalk, do not engage: politely refuse and explain RAG scope
        return (
            "I’m a retrieval‑augmented assistant focused on answering questions using your "
            "uploaded PDFs. Please ask a knowledge‑based question (or provide documents) "
            "so I can help with citations."
        )

    header = (
        "You are a factual assistant. Use ONLY the provided context to answer. "
        "Cite evidence by chunk index when helpful. If insufficient, say 'insufficient evidence'.\n\n"
    )
    ctx = "\n\n".join([f"[Chunk {i}]\n{t}" for i, t in enumerate(context_chunks, 1)])

    if mode == "list":
        instruction = (
            "Return a JSON array of items directly supported by the context. No extra commentary."
        )
    elif mode == "table":
        instruction = (
            "Return a JSON array of objects (table rows) with consistent keys, using only supported facts."
        )
    else:
        instruction = "Answer concisely with citations."

    return f"{header}Context:\n{ctx}\n\nQuestion: {query}\n\nInstruction: {instruction}"


