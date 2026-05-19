"""
Chat route — post-report conversational AI endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.risk_queue import job_queue, JobStatus
from inference.chat_engine import chat

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    job_id: str
    message: str
    history: list[dict] = []


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """Answer a user question about a completed fraud report."""
    job = job_queue.get(req.job_id)
    if not job:
        raise HTTPException(404, "Report not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(400, "Report is not yet complete.")

    response = await chat(
        user_message=req.message,
        report_summary=job.result or {},
        conversation_history=req.history,
    )

    return JSONResponse({"reply": response})
