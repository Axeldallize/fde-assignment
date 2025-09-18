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

    # LLM/Embeddings provider (Mistral by default)
    llm_provider: str = os.getenv("LLM_PROVIDER", "mistral")
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "mistral")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "mistral-embed")


settings = Settings()

