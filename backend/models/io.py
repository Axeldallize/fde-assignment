from pydantic import BaseModel
from typing import List, Optional, Literal


class IngestResponse(BaseModel):
    ingested: List[str]
    chunks: int
    warnings: List[str] = []


class QueryRequest(BaseModel):
    query: str
    mode: Optional[Literal["auto", "qa", "list", "table"]] = "auto"
    top_k: int = 12
    semantic: bool = True
    llm_expand: bool = False
    # Runtime overrides
    use_rrf: Optional[bool] = None
    evidence_threshold: Optional[float] = None
    evidence_topk: Optional[int] = None
    temperature: Optional[float] = None


class Citation(BaseModel):
    doc_id: str
    pages: str
    heading: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    citations: List[Citation] = []
    meta: dict = {}
    error: Optional[str] = None
    reason: Optional[str] = None

