"""
Upload route — accepts any CSV or XLSX file, starts an async analysis job,
and provides a polling endpoint for live progress updates.
"""
from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from config import UPLOAD_DIR, MAX_UPLOAD_MB, LLM_MAX_ACCOUNTS_PER_JOB, ONNX_MODEL_PATH, ML_FLAG_THRESHOLD
from core.schema_detector import load_file, detect_schema, ROLE_ACCOUNT_ID
from core.detector import FraudDetector
from core.risk_queue import job_queue, JobStatus
from inference.llm_engine import llm_engine
from inference.prompt_builder import build_prompt
from inference.response_parser import parse_response

router = APIRouter(prefix="/api")

_detector = FraudDetector()


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Accept a CSV or XLSX file and kick off background analysis."""
    # Validate file type
    name = file.filename or ""
    if not (name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".xls")):
        raise HTTPException(400, "Only CSV and XLSX files are supported.")

    # Read content
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_MB} MB.")

    # Save to disk
    dest = UPLOAD_DIR / name
    dest.write_bytes(content)

    # Create job
    job = job_queue.create_job(name)
    job_queue.update(job.id, 0.01, "File received — starting analysis…", JobStatus.RUNNING)

    # Kick off background analysis
    background_tasks.add_task(_run_analysis, job.id, str(dest), name)

    return JSONResponse({"job_id": job.id, "filename": name})


@router.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Poll the status and progress of an analysis job."""
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return JSONResponse(job.to_dict())


@router.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the final report once a job is complete."""
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(202, "Job not yet complete.")
    return JSONResponse(job.result)


# ─── Background Analysis Pipeline ───────────────────────────────────────────────

async def _run_analysis(job_id: str, file_path: str, filename: str) -> None:
    """Full pipeline: load → detect schema → ML score → LLM deep dive → build report."""
    cb = job_queue.make_progress_callback(job_id)
    try:
        # Load file
        cb(0.02, "Reading file…")
        df = await asyncio.get_event_loop().run_in_executor(None, load_file, file_path)

        cb(0.05, f"Loaded {len(df):,} rows — detecting schema…")
        mapping = detect_schema(df)

        cb(0.10, f"Schema detected ({len(mapping.role_map)} roles found) — running ML screening…")

        # ML detection
        txn_scores = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _detector.fit_and_score(df, mapping, cb),
        )

        cb(0.65, "Aggregating account risk scores…")
        account_summary = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _detector.aggregate_accounts(txn_scores),
        )

        # LLM deep dive on flagged accounts
        flagged_accounts = account_summary[
            account_summary["account_risk_score"] >= ML_FLAG_THRESHOLD
        ].head(LLM_MAX_ACCOUNTS_PER_JOB)

        total_flagged = len(flagged_accounts)
        llm_results: list[dict] = []

        for i, (_, row) in enumerate(flagged_accounts.iterrows()):
            pct = 0.67 + (i / max(total_flagged, 1)) * 0.25
            acct_id = row["account_id"]
            cb(pct, f"Analyzing account {i+1}/{total_flagged}: {acct_id}…")

            # Get this account's transactions
            if mapping.has(ROLE_ACCOUNT_ID):
                acct_df = df[df[mapping.get(ROLE_ACCOUNT_ID)].astype(str) == acct_id]
            else:
                acct_df = df

            prompt = build_prompt(
                account_id=acct_id,
                account_risk_score=row["account_risk_score"],
                ml_reasons=row["top_reasons"],
                transactions=acct_df,
                mapping=mapping,
                total_accounts=account_summary["account_id"].nunique(),
                flagged_accounts=total_flagged,
                account_rank=i + 1,
            )

            raw_response = await llm_engine.generate(prompt)
            verdict = parse_response(raw_response)

            llm_results.append({
                "account_id": acct_id,
                "ml_risk_score": float(row["account_risk_score"]),
                "flagged_txn_count": int(row["flagged_txn_count"]),
                "total_txn_count": int(row["total_txn_count"]),
                "top_reasons": row["top_reasons"],
                **verdict.to_dict(),
            })

        cb(0.93, "Building final report…")

        # Build report
        total_txns = len(df)
        total_accounts_count = account_summary["account_id"].nunique()
        flagged_txns = int(txn_scores["is_flagged"].sum())
        fraud_pct = (flagged_txns / max(total_txns, 1)) * 100

        report = {
            "job_id": job_id,
            "filename": filename,
            "schema_summary": mapping.summary(),
            "overview": {
                "total_transactions": total_txns,
                "total_accounts": total_accounts_count,
                "flagged_transactions": flagged_txns,
                "fraud_percentage": round(fraud_pct, 1),
                "high_risk_count": len([a for a in llm_results if a["verdict"] in ("CRITICAL", "HIGH")]),
                "str_recommended": len([a for a in llm_results if a["action"] == "FILE_STR"]),
            },
            "flagged_accounts": sorted(llm_results, key=lambda x: x["ml_risk_score"], reverse=True),
        }

        job_queue.complete(job_id, report)

    except Exception as exc:
        job_queue.fail(job_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
