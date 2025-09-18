from fastapi import FastAPI
from .config import settings
from .utils.logging import configure_logging


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

