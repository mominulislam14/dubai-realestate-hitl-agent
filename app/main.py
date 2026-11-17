"""
FastAPI application entrypoint for DubaiProp HITL.
Manages async SQLite checkpointer lifecycle, Qdrant warm-up, and static dashboard serving.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.api.routes import router as api_router
from app.core.config import settings
from app.graph.workflow import build_workflow
from app.rag.vector_store import ensure_kb_ready, get_qdrant_client

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vector store and persistent LangGraph checkpointer on startup."""
    ensure_kb_ready()
    workflow = build_workflow()

    async with AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH) as checkpointer:
        await checkpointer.setup()
        app.state.graph = workflow.compile(checkpointer=checkpointer)
        try:
            yield
        finally:
            get_qdrant_client().close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Autonomous Business Inquiry & Deal Agent with Corrective RAG and Human-in-the-Loop (HITL).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the DubaiProp HITL visual dashboard if present."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        {
            "app": settings.APP_NAME,
            "status": "running",
            "docs": "/docs",
            "health": "/api/health",
        }
    )