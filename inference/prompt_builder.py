"""
Prompt Builder — constructs chain-of-thought prompts for Phi-3.5-mini.

Supports two model families:
  'phi35' — Microsoft Phi-3.5-mini-instruct (<|user|>/<|assistant|> template)
  'qwen'  — Qwen2.5-7B (<|im_start|>user/<|im_end|> ChatML template)

Also injects dynamically selected few-shot examples from fraud_few_shot.py
so the model behaves like a trained fraud expert even before fine-tuning.
"""
from __future__ import annotations

import textwrap
from typing import Optional

import pandas as pd

from core.schema_detector import (
    ColumnMapping, ROLE_ACCOUNT_ID, ROLE_AMOUNT, ROLE_TIMESTAMP,
    ROLE_DESCRIPTION, ROLE_MERCHANT,
)
from core.feature_engineer import _parse_amount
from inference.fraud_few_shot import select_examples, format_few_shot_block
from config import MODEL_FAMILY


_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert AML (Anti-Money Laundering) analyst specializing in
    transaction fraud detection for fintech platforms. You analyze account
    activity and provide structured risk assessments.

    You must respond in EXACTLY this format — no extra text before or after:
    VERDICT: [CRITICAL|HIGH|MEDIUM|LOW]
    CONFIDENCE: [0-100]%
    ACTION: [FILE_STR|ESCALATE|MONITOR|CLEAR]
    ANALYSIS: [2-3 sentence plain-English explanation of your reasoning]
    STR_NARRATIVE: [1-2 sentence FINTRAC-style narrative, or "N/A" if no STR needed]

    Fraud typologies to check for:
    - Structuring: multiple transactions just below reporting thresholds ($10,000 CAD)
    - Smurfing: splitting large amounts across many small transactions from multiple sources
    - Layering: rapid transfers between accounts to obscure origin (receive-and-forward)
    - Dormant activation: sudden high activity on previously dormant account
    - Mule accounts: account receives and immediately forwards funds to other accounts

    Key thresholds (Canada/FINTRAC):
    - $10,000 CAD: mandatory cash transaction report (CTR) threshold
    - $7,500–$9,999 CAD: structuring danger zone
    - 15+ transactions/day: high velocity flag
    - Off-hours (22:00–06:00 local): elevated risk indicator
""").strip()


def _build_phi35_prompt(
    system: str,
    few_shot_block: str,
    account_context: str,
) -> str:
    """Phi-3.5-mini chat template: <|system|>...<|user|>...<|assistant|>"""
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"{few_shot_block}"
        f"<|user|>\n{account_context}<|end|>\n"
        f"<|assistant|>\n"
    )


def _build_qwen_prompt(
    system: str,
    few_shot_block: str,
    account_context: str,
) -> str:
    """Qwen2.5 ChatML template: <|im_start|>...<|im_end|>"""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"{few_shot_block}"
        f"<|im_start|>user\n{account_context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_prompt(
    account_id: str,
    account_risk_score: float,
    ml_reasons: list[str],
    transactions: pd.DataFrame,
    mapping: ColumnMapping,
    total_accounts: int,
    flagged_accounts: int,
    account_rank: int = 1,
    model_family: Optional[str] = None,
) -> str:
    """Build a full chain-of-thought prompt for a single account."""

    if model_family is None:
        model_family = MODEL_FAMILY

    # ── Amount statistics ────────────────────────────────────────────────────────
    amount_summary   = "Amount data not available"
    structuring_note = ""
    if mapping.has(ROLE_AMOUNT):
        amounts     = _parse_amount(transactions[mapping.get(ROLE_AMOUNT)])
        total_vol   = amounts.abs().sum()
        max_single  = amounts.abs().max()
        avg_txn     = amounts.abs().mean()
        amount_summary = (
            f"Total volume: ${total_vol:,.2f} CAD | "
            f"Largest single txn: ${max_single:,.2f} | "
            f"Average txn: ${avg_txn:,.2f}"
        )
        structuring_txns = amounts[(amounts.abs() >= 7500) & (amounts.abs() < 10000)]
        if len(structuring_txns) > 0:
            structuring_note = (
                f"\n  ⚠️  STRUCTURING ALERT: {len(structuring_txns)} transaction(s) in "
                f"$7,500–$9,999 range (just below $10,000 FINTRAC reporting threshold). "
                f"Total structured amount: ${structuring_txns.abs().sum():,.2f} CAD"
            )

    # ── Time span analysis ───────────────────────────────────────────────────────
    date_range_str = ""
    velocity_str   = ""
    if mapping.has(ROLE_TIMESTAMP):
        ts = pd.to_datetime(transactions[mapping.get(ROLE_TIMESTAMP)], errors="coerce").dropna()
        if len(ts) >= 2:
            span_days      = max((ts.max() - ts.min()).days, 1)
            daily_rate     = len(transactions) / span_days
            date_range_str = f"Activity span: {ts.min().date()} to {ts.max().date()} ({span_days} days)"
            velocity_str   = f"Average velocity: {daily_rate:.1f} transactions/day"

    # ── Format transaction table ─────────────────────────────────────────────────
    txn_table = _format_transactions(transactions, mapping, max_rows=25)

    # ── Few-shot injection ───────────────────────────────────────────────────────
    examples         = select_examples(ml_reasons, n=3)
    few_shot_block   = format_few_shot_block(examples, model_family=model_family)

    # ── Assemble account context (the "user" turn) ───────────────────────────────
    account_context = textwrap.dedent(f"""
        ACCOUNT UNDER INVESTIGATION
        Account ID: {account_id}
        ML Risk Score: {account_risk_score:.1%}  (rank #{account_rank} of {flagged_accounts} flagged accounts)
        Total Transactions: {len(transactions)}
        {amount_summary}{structuring_note}
        {date_range_str}
        {velocity_str}
        ML-Detected Anomalies: {', '.join(ml_reasons) if ml_reasons else 'None detected by rule engine'}

        PORTFOLIO CONTEXT
        Account #{account_rank} of {flagged_accounts} flagged / {total_accounts} total accounts in batch.

        TRANSACTION HISTORY (up to 25 most recent):
        {txn_table}

        ---
        Step 1: Examine ALL transactions above for suspicious patterns.
        Step 2: Check specifically for: structuring, smurfing, layering, dormant activation, mule behaviour.
        Step 3: Consider velocity, amounts, timing, and counterparties together.
        Step 4: Weigh the ML risk score ({account_risk_score:.1%}) as one signal among many.
        Step 5: Assign your VERDICT and ACTION based on the full picture.
    """).strip()

    # ── Build final prompt in the correct chat template ──────────────────────────
    if model_family == "phi35":
        return _build_phi35_prompt(_SYSTEM_PROMPT, few_shot_block, account_context)
    else:
        return _build_qwen_prompt(_SYSTEM_PROMPT, few_shot_block, account_context)


def _format_transactions(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    max_rows: int = 25,
) -> str:
    """Format transaction rows as a Markdown table using canonical role names."""
    display_cols: list[tuple[str, str]] = []

    role_display = [
        (ROLE_TIMESTAMP,   "Date/Time"),
        (ROLE_AMOUNT,      "Amount"),
        (ROLE_DESCRIPTION, "Description"),
        (ROLE_MERCHANT,    "Merchant"),
    ]
    for role, name in role_display:
        if mapping.has(role):
            display_cols.append((mapping.get(role), name))

    # Include up to 2 unmapped columns for additional context
    extra = 0
    for col in mapping.unmapped:
        if extra >= 2:
            break
        display_cols.append((col, col))
        extra += 1

    if not display_cols:
        return "_No displayable columns detected._"

    sample       = df.tail(max_rows).copy()
    raw_cols     = [c for c, _ in display_cols]
    display_names = [n for _, n in display_cols]

    rows   = []
    header = "| " + " | ".join(display_names) + " |"
    sep    = "|" + "|".join(["---"] * len(display_names)) + "|"
    rows.append(header)
    rows.append(sep)

    for _, row in sample[raw_cols].iterrows():
        cells = []
        for col in raw_cols:
            val = str(row[col])
            if len(val) > 40:
                val = val[:37] + "…"
            cells.append(val)
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)
