"""
Schema Detector — auto-discovers column meanings from any CSV/XLSX file.

Uses heuristic keyword matching + statistical profiling to map unknown
column names to canonical roles: account_id, amount, timestamp, etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─── Canonical Field Roles ──────────────────────────────────────────────────────
ROLE_ACCOUNT_ID    = "account_id"
ROLE_AMOUNT        = "amount"
ROLE_TIMESTAMP     = "timestamp"
ROLE_DESCRIPTION   = "description"
ROLE_MERCHANT      = "merchant"
ROLE_CATEGORY      = "category"
ROLE_DIRECTION     = "direction"      # debit / credit
ROLE_COUNTERPARTY  = "counterparty"
ROLE_BALANCE       = "balance"
ROLE_CURRENCY      = "currency"
ROLE_STATUS        = "status"
ROLE_TRANSACTION_ID = "transaction_id"

# Keywords that strongly suggest each role
_ROLE_HINTS: dict[str, list[str]] = {
    ROLE_ACCOUNT_ID:    ["account", "acct", "acc", "user", "client", "customer", "member", "userid", "holder"],
    ROLE_TRANSACTION_ID:["txn", "transaction", "trans", "tran", "ref", "reference", "id", "uuid"],
    ROLE_AMOUNT:        ["amount", "amt", "value", "sum", "total", "price", "charge", "payment", "debit", "credit", "money"],
    ROLE_TIMESTAMP:     ["date", "time", "datetime", "timestamp", "created", "posted", "processed", "when", "at"],
    ROLE_DESCRIPTION:   ["description", "desc", "note", "memo", "narrative", "detail", "remark", "comment"],
    ROLE_MERCHANT:      ["merchant", "vendor", "store", "shop", "payee", "recipient", "seller", "company"],
    ROLE_CATEGORY:      ["category", "cat", "type", "kind", "tag", "label", "class", "segment"],
    ROLE_DIRECTION:     ["direction", "dir", "type", "dr_cr", "drcr", "indicator", "sign"],
    ROLE_COUNTERPARTY:  ["counterparty", "counter", "other", "party", "sender", "receiver", "from", "to"],
    ROLE_BALANCE:       ["balance", "bal", "running", "available", "remaining"],
    ROLE_CURRENCY:      ["currency", "ccy", "curr", "iso"],
    ROLE_STATUS:        ["status", "state", "result", "outcome", "flag", "approved", "pending"],
}


@dataclass
class ColumnMapping:
    """Maps detected column names to canonical roles."""
    raw_columns: list[str]
    role_map: dict[str, str] = field(default_factory=dict)   # role → raw column name
    unmapped:  list[str]     = field(default_factory=list)

    def get(self, role: str) -> Optional[str]:
        return self.role_map.get(role)

    def has(self, role: str) -> bool:
        return role in self.role_map

    def summary(self) -> dict:
        return {
            "detected_roles": list(self.role_map.keys()),
            "unmapped_columns": self.unmapped,
            "total_columns": len(self.raw_columns),
        }


def _normalize(name: str) -> str:
    """Lowercase, strip whitespace and non-alphanumeric chars."""
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


def _score_column(col_norm: str, hints: list[str]) -> int:
    """Return how many hint keywords appear inside the column name."""
    return sum(1 for h in hints if h in col_norm)


def detect_schema(df: pd.DataFrame) -> ColumnMapping:
    """
    Auto-detect column roles from a DataFrame.

    Strategy:
    1. Keyword scoring against hint lists.
    2. Statistical fallback: if no amount column found, pick the most
       numeric column; if no timestamp found, pick the most date-like.
    3. Each role is assigned at most once (greedy best-match).
    """
    raw_cols = list(df.columns)
    normalized = {col: _normalize(col) for col in raw_cols}

    # Build a score matrix: role → {col: score}
    scores: dict[str, dict[str, int]] = {}
    for role, hints in _ROLE_HINTS.items():
        scores[role] = {col: _score_column(normalized[col], hints) for col in raw_cols}

    assigned: dict[str, str] = {}   # role → col
    used_cols: set[str] = set()

    # Greedy assignment: highest score wins, break ties by column order
    roles_priority = [
        ROLE_ACCOUNT_ID, ROLE_TRANSACTION_ID, ROLE_AMOUNT,
        ROLE_TIMESTAMP, ROLE_DESCRIPTION, ROLE_MERCHANT,
        ROLE_CATEGORY, ROLE_DIRECTION, ROLE_COUNTERPARTY,
        ROLE_BALANCE, ROLE_CURRENCY, ROLE_STATUS,
    ]

    for role in roles_priority:
        best_col = None
        best_score = 0
        for col in raw_cols:
            if col in used_cols:
                continue
            s = scores[role][col]
            if s > best_score:
                best_score = s
                best_col = col
        if best_col and best_score > 0:
            assigned[role] = best_col
            used_cols.add(best_col)

    # Statistical fallback for critical missing roles
    if ROLE_AMOUNT not in assigned:
        # Find the most numeric column not yet used
        for col in raw_cols:
            if col in used_cols:
                continue
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().mean() > 0.7:
                    assigned[ROLE_AMOUNT] = col
                    used_cols.add(col)
                    break
            except Exception:
                pass

    if ROLE_TIMESTAMP not in assigned:
        for col in raw_cols:
            if col in used_cols:
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    assigned[ROLE_TIMESTAMP] = col
                    used_cols.add(col)
                    break
            except Exception:
                pass

    if ROLE_ACCOUNT_ID not in assigned:
        # Pick the first string column that looks like IDs (high cardinality)
        for col in raw_cols:
            if col in used_cols:
                continue
            if df[col].dtype == object:
                unique_ratio = df[col].nunique() / max(len(df), 1)
                if unique_ratio > 0.05:
                    assigned[ROLE_ACCOUNT_ID] = col
                    used_cols.add(col)
                    break

    unmapped = [c for c in raw_cols if c not in used_cols]
    return ColumnMapping(raw_columns=raw_cols, role_map=assigned, unmapped=unmapped)


def load_file(path: str) -> pd.DataFrame:
    """Load a CSV or XLSX file into a DataFrame regardless of format."""
    path_lower = path.lower()
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        return pd.read_excel(path, sheet_name=0, dtype=str)
    else:
        # Try different encodings / separators
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(path, dtype=str, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not read CSV file with common encodings: {path}")
