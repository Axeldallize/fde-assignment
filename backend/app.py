from fastapi import FastAPI, UploadFile, File
from .config import settings
from .utils.logging import configure_logging
from .models.io import IngestResponse
from .ingestion.service import ingest_files
from pathlib import Path


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

