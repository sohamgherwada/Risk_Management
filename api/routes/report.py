"""
Report route — retrieval and PDF export.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

from core.risk_queue import job_queue, JobStatus
from pdf.report_renderer import render_pdf

router = APIRouter(prefix="/api")


@router.get("/report/{job_id}/pdf")
async def download_pdf(job_id: str):
    """Generate and return the report as a downloadable PDF."""
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Report not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(400, "Report is not yet complete.")

    pdf_bytes = render_pdf(job.result)
    filename = f"fraud_report_{job_id[:8]}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
