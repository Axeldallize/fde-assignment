import os
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseModel):
    app_name: str = "RAG Pipeline API"
    env: str = os.getenv("ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Retrieval flags
    use_semantic: bool = os.getenv("USE_SEMANTIC", "true").lower() == "true"
    use_rrf: bool = os.getenv("USE_RRF", "false").lower() == "true"  # default FALSE per decision

    # Evidence thresholds
    evidence_topk: int = int(os.getenv("EVIDENCE_TOPK", "4"))
    evidence_threshold: float = float(os.getenv("EVIDENCE_THRESHOLD", "0.28"))

    # LLM provider (Anthropic by default)
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Embeddings provider (Voyage by default)
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "voyage")
    voyage_api_key: str | None = os.getenv("VOYAGE_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "voyage-3.5")


settings = Settings()

