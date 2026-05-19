"""
Feature Engineer — converts schema-detected DataFrame rows into
numeric feature vectors suitable for IsolationForest / XGBoost.

Works entirely on the canonical ColumnMapping, never on raw column names.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

import numpy as np
import pandas as pd

from core.schema_detector import (
    ColumnMapping,
    ROLE_ACCOUNT_ID, ROLE_AMOUNT, ROLE_TIMESTAMP, ROLE_DESCRIPTION,
    ROLE_MERCHANT, ROLE_DIRECTION, ROLE_COUNTERPARTY, ROLE_BALANCE,
    ROLE_STATUS,
)

# Keywords that appear in fraudulent/suspicious descriptions
_SUSPICIOUS_KEYWORDS = [
    "urgent", "lottery", "prize", "wire", "western union", "bitcoin",
    "crypto", "gift card", "refund", "overpayment", "investment",
    "guaranteed", "cash", "offshore", "anonymous", "untraceable",
]

# Hours considered "off-hours" (higher risk)
_OFF_HOURS = set(range(0, 6)) | {23}


def _parse_amount(series: pd.Series) -> pd.Series:
    """Coerce any amount column to float, handling currency symbols."""
    return (
        series.astype(str)
        .str.replace(r"[$,£€¥\s]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (100) → -100
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def _suspicious_text_score(text: str) -> float:
    """Return fraction of suspicious keywords present in text."""
    if not isinstance(text, str) or not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in _SUSPICIOUS_KEYWORDS if kw in text_lower)
    return min(hits / 3.0, 1.0)


def engineer_features(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """
    Build a feature DataFrame from a raw DataFrame using the column mapping.

    Returns a DataFrame of numeric features aligned with `df` index.
    """
    feat: dict[str, pd.Series] = {}
    n = len(df)

    # ── Amount features ─────────────────────────────────────────────────────────
    if mapping.has(ROLE_AMOUNT):
        amounts = _parse_amount(df[mapping.get(ROLE_AMOUNT)])
        feat["amount"] = amounts
        feat["amount_abs"] = amounts.abs()
        feat["amount_log"] = np.log1p(amounts.abs())
        feat["amount_is_round"] = (amounts.abs() % 100 == 0).astype(float)
        feat["amount_is_large"] = (amounts.abs() > 5000).astype(float)
    else:
        for k in ["amount", "amount_abs", "amount_log", "amount_is_round", "amount_is_large"]:
            feat[k] = pd.Series(np.zeros(n))

    # ── Timestamp features ──────────────────────────────────────────────────────
    if mapping.has(ROLE_TIMESTAMP):
        ts = pd.to_datetime(df[mapping.get(ROLE_TIMESTAMP)], errors="coerce")
        feat["hour"] = ts.dt.hour.fillna(12).astype(float)
        feat["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(float)
        feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(float)
        feat["is_off_hours"] = feat["hour"].apply(lambda h: 1.0 if int(h) in _OFF_HOURS else 0.0)
    else:
        feat["hour"] = pd.Series(np.full(n, 12.0))
        feat["day_of_week"] = pd.Series(np.zeros(n))
        feat["is_weekend"] = pd.Series(np.zeros(n))
        feat["is_off_hours"] = pd.Series(np.zeros(n))

    # ── Text / description features ──────────────────────────────────────────────
    desc_col = mapping.get(ROLE_DESCRIPTION) or mapping.get(ROLE_MERCHANT)
    if desc_col:
        feat["suspicious_text"] = df[desc_col].apply(_suspicious_text_score)
        feat["desc_length"] = df[desc_col].astype(str).str.len().fillna(0).astype(float)
    else:
        feat["suspicious_text"] = pd.Series(np.zeros(n))
        feat["desc_length"] = pd.Series(np.zeros(n))

    # ── Account-level velocity features ─────────────────────────────────────────
    if mapping.has(ROLE_ACCOUNT_ID):
        acct = df[mapping.get(ROLE_ACCOUNT_ID)].astype(str)
        txn_count = acct.map(acct.value_counts())
        feat["account_txn_count"] = txn_count.fillna(1).astype(float)

        if mapping.has(ROLE_AMOUNT):
            acct_total = acct.map(
                df.groupby(acct)[mapping.get(ROLE_AMOUNT)]
                .apply(lambda s: _parse_amount(s).abs().sum())
            )
            feat["account_total_volume"] = acct_total.fillna(0).astype(float)
        else:
            feat["account_total_volume"] = pd.Series(np.zeros(n))
    else:
        feat["account_txn_count"] = pd.Series(np.ones(n))
        feat["account_total_volume"] = pd.Series(np.zeros(n))

    # ── Balance deviation ────────────────────────────────────────────────────────
    if mapping.has(ROLE_BALANCE):
        bal = _parse_amount(df[mapping.get(ROLE_BALANCE)])
        feat["balance"] = bal
        feat["balance_negative"] = (bal < 0).astype(float)
    else:
        feat["balance"] = pd.Series(np.zeros(n))
        feat["balance_negative"] = pd.Series(np.zeros(n))

    return pd.DataFrame(feat)
