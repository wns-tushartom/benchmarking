# type: ignore
import os
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Resolve absolute path to the `app` package (for static dir detection)
import source as app_pkg

# ---- Config & Routers ----
from source.routers import documents, search, llm, embeddings, query_analyzer
from source.models.schemas import HealthResponse, ServiceStatus, SystemStats


from source.config import load_cloud_services
cloud_services = load_cloud_services()
# ============================= Logging (safe defaults) =============================
DEFAULT_LOG_LEVEL = cloud_services["log_level"]
_level_name = str(getattr(cloud_services, "log_level", DEFAULT_LOG_LEVEL)).upper()
LOG_LEVEL = getattr(logging, _level_name, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _cfg(name: str, default):
    """Read value from settings with a safe fallback."""
    return getattr(cloud_services, name, default)


# ================================= Lifespan =================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("WNS AI Research Retreiver System starting up...")

    try:
        # Importing ensures singletons initialize and log their status
        from source.services.vector_service import get_vector_service  # noqa: F401
        vector_service = get_vector_service()  # Initialize the service
        from source.services.embedding_service import embedding_service  # noqa: F401
        from source.services.llm_service import llm_service  # noqa: F401
        logger.info("Core services import OK")
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
    yield
    logger.info("System shutting down...")


# ================================== App ==================================
app = FastAPI(
    title="Retreiver System",
    description="Retreiver with document parsing, embeddings, Milvus vector search, Query Analyzer and LLM calling.",
    version="1.0.0",
    lifespan=lifespan,
)

# ================================= Routers =================================
app.include_router(documents.router)
app.include_router(embeddings.router)
app.include_router(search.router)
app.include_router(query_analyzer.router)
app.include_router(llm.router)


    
# ================================ Entrypoint ================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=_cfg("api_host", "127.0.0.1"),
        port=int(_cfg("api_port", 8000)),
        reload=True,
        log_level=str(getattr(cloud_services, "log_level", DEFAULT_LOG_LEVEL)).lower(),
    )
