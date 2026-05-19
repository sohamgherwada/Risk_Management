"""
FastAPI main entrypoint — serves both the REST API and the static dashboard.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import HOST, PORT, DEBUG, ONNX_MODEL_PATH
from api.routes.upload import router as upload_router
from api.routes.chat import router as chat_router
from api.routes.report import router as report_router
from inference.llm_engine import llm_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

app = FastAPI(
    title="NLP Risk Monitoring System",
    description="AI-powered fraud detection for fintech — upload any CSV/XLSX, get instant risk analysis.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(report_router)


@app.on_event("startup")
async def startup():
    logger.info("Starting NLP Risk Monitoring System…")
    loaded = await llm_engine.load(ONNX_MODEL_PATH)
    if loaded:
        logger.info("✅ Phi-3.5-mini ONNX INT4 loaded (CUDA) — ready for fraud analysis")
    else:
        logger.warning("⚠️  LLM model not loaded — using intelligent mock responses")


@app.get("/health")
async def health():
    return {
        "status":              "ok",
        "llm_loaded":         llm_engine._loaded,
        "llm_mode":           "pytorch-cuda" if llm_engine._loaded else "mock",
        "pq_avg_ratio":       round(llm_engine.pq_average_ratio, 2),
        "pq_calls":           llm_engine._pq_calls,
    }


# ─── Serve dashboard static files ───────────────────────────────────────────────
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))

    @app.get("/processing")
    async def processing():
        return FileResponse(str(DASHBOARD_DIR / "processing.html"))

    @app.get("/report")
    async def report():
        return FileResponse(str(DASHBOARD_DIR / "report.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=DEBUG)
