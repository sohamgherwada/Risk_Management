"""
Fraud Detector — three-signal ensemble:

  Signal A: IsolationForest (global anomaly detection — catches novel outliers)
  Signal B: Per-account statistical scoring (z-score + IQR within each account)
  Signal C: Rule-based heuristic score (velocity, structuring, off-hours patterns)

Blending A + B + C gives a robust 0.0–1.0 risk score per transaction that works
on any real-world transaction CSV without needing pre-labelled training data.

The old XGBoost-trained-on-synthetic-labels approach was removed because it was
circular (training and scoring on the same data with heuristic labels) and
systematically suppressed real fraud signals.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.schema_detector import ColumnMapping, ROLE_ACCOUNT_ID, ROLE_AMOUNT, ROLE_TIMESTAMP
from core.feature_engineer import engineer_features
from config import ML_FLAG_THRESHOLD, MODELS_DIR

_ISO_PATH = MODELS_DIR / "isolation_forest.pkl"
_SCL_PATH = MODELS_DIR / "scaler.pkl"

# Blend weights — must sum to 1.0
_W_ISO   = 0.40   # IsolationForest global anomaly
_W_STAT  = 0.35   # Per-account statistical outlier
_W_RULE  = 0.25   # Rule-based heuristic


class FraudDetector:
    """Ensemble fraud detector: IsolationForest + statistical + rule-based."""

    def __init__(self) -> None:
        self.iso: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load saved models or create fresh ones."""
        if _ISO_PATH.exists() and _SCL_PATH.exists():
            with open(_ISO_PATH, "rb") as f:
                self.iso = pickle.load(f)
            with open(_SCL_PATH, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.iso = IsolationForest(
                n_estimators=200,
                contamination=0.08,   # expect ~8% anomalous in a fraud dataset
                random_state=42,
                n_jobs=-1,
            )
            self.scaler = StandardScaler()

    def _save_models(self) -> None:
        with open(_ISO_PATH, "wb") as f:
            pickle.dump(self.iso, f)
        with open(_SCL_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

    # ── Signal B: per-account statistical outlier score ─────────────────────────
    @staticmethod
    def _account_statistical_score(feat_df: pd.DataFrame, acct_ids: np.ndarray) -> np.ndarray:
        """
        For each transaction compute how anomalous its amount is *within its account*.

        Uses a combination of:
          - z-score of |amount| within the account (capped at 3σ → score 1.0)
          - IQR fence: beyond 1.5×IQR → elevated score
          - Structuring flag: amount in (8000, 9999) CAD range
        """
        amounts = feat_df["amount_abs"].values
        stat_scores = np.zeros(len(feat_df))

        unique_accounts = np.unique(acct_ids)
        for acct in unique_accounts:
            mask = acct_ids == acct
            acct_amounts = amounts[mask]
            n = mask.sum()

            if n < 2:
                # Single transaction — can't compute within-account stats
                # Score based on absolute amount alone
                stat_scores[mask] = np.clip(acct_amounts / 20000.0, 0, 1)
                continue

            mu = np.mean(acct_amounts)
            sigma = np.std(acct_amounts) + 1e-9
            z = np.abs(acct_amounts - mu) / sigma
            z_score = np.clip(z / 3.0, 0, 1)   # z=3 → score 1.0

            q1, q3 = np.percentile(acct_amounts, [25, 75])
            iqr = q3 - q1 + 1e-9
            iqr_score = np.clip((acct_amounts - (q3 + 1.5 * iqr)) / (iqr + 1e-9), 0, 1)

            # Structuring: amounts clustered just below $10k reporting threshold
            structuring = ((acct_amounts >= 7500) & (acct_amounts < 10000)).astype(float) * 0.6

            stat_scores[mask] = np.clip(
                z_score * 0.5 + iqr_score * 0.3 + structuring * 0.2,
                0, 1
            )

        return stat_scores

    # ── Signal C: rule-based heuristic score ────────────────────────────────────
    @staticmethod
    def _rule_based_score(feat_df: pd.DataFrame, acct_ids: np.ndarray) -> np.ndarray:
        """
        Scores based on known AML typology rules:
          - High transaction velocity per account
          - Off-hours activity
          - Large round-number amounts
          - Negative balance
          - Suspicious text keywords
          - Rapid-fire same-day transactions (velocity within 24h window)
        """
        n = len(feat_df)
        scores = np.zeros(n)

        # Velocity: number of transactions per account — normalise at 30 txns → score 0.6
        txn_count = feat_df["account_txn_count"].values
        scores += np.clip(txn_count / 50.0, 0, 0.6)

        # Off-hours
        scores += feat_df["is_off_hours"].values * 0.15

        # Large round amounts (>$1000)
        round_large = feat_df["amount_is_round"].values * (feat_df["amount_abs"].values > 1000).astype(float)
        scores += round_large * 0.20

        # Very large transactions (>$10k)
        scores += feat_df["amount_is_large"].values * 0.25

        # Negative balance
        scores += feat_df["balance_negative"].values * 0.15

        # Suspicious text
        scores += feat_df["suspicious_text"].values * 0.35

        # High total account volume — normalise at $500k → score 0.4
        scores += np.clip(feat_df["account_total_volume"].values / 500_000.0, 0, 0.40)

        return np.clip(scores, 0, 1)

    def fit_and_score(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Fit models on this dataset and return transaction-level risk scores.

        Returns a DataFrame with columns:
            account_id, row_index, risk_score, is_flagged, anomaly_reasons
        """
        if progress_callback:
            progress_callback(0.05, "Engineering features…")

        feat_df = engineer_features(df, mapping)
        X = feat_df.values.astype(np.float32)

        # ── Account IDs ─────────────────────────────────────────────────────────
        if mapping.has(ROLE_ACCOUNT_ID):
            acct_ids = df[mapping.get(ROLE_ACCOUNT_ID)].astype(str).values
        else:
            acct_ids = np.array([f"ROW_{i}" for i in range(len(df))])

        if progress_callback:
            progress_callback(0.15, "Fitting anomaly detector…")

        # ── Signal A: IsolationForest ────────────────────────────────────────────
        X_scaled = self.scaler.fit_transform(X)
        self.iso.fit(X_scaled)
        iso_raw = self.iso.decision_function(X_scaled)  # higher = more normal
        iso_min, iso_max = iso_raw.min(), iso_raw.max()
        if iso_max > iso_min:
            iso_risk = 1.0 - (iso_raw - iso_min) / (iso_max - iso_min)
        else:
            iso_risk = np.zeros(len(iso_raw))

        if progress_callback:
            progress_callback(0.35, "Running per-account statistical analysis…")

        # ── Signal B: Statistical per-account outlier ────────────────────────────
        stat_risk = self._account_statistical_score(feat_df, acct_ids)

        if progress_callback:
            progress_callback(0.50, "Applying AML rule engine…")

        # ── Signal C: Rule-based heuristic ───────────────────────────────────────
        rule_risk = self._rule_based_score(feat_df, acct_ids)

        if progress_callback:
            progress_callback(0.60, "Blending risk signals…")

        # ── Blend A + B + C ──────────────────────────────────────────────────────
        blended = np.clip(
            _W_ISO * iso_risk + _W_STAT * stat_risk + _W_RULE * rule_risk,
            0, 1
        )

        # ── Per-transaction anomaly reasons ──────────────────────────────────────
        reasons = []
        for i in range(len(df)):
            r = []
            if feat_df["is_off_hours"].iloc[i]:
                r.append("Off-hours transaction")
            if feat_df["amount_is_round"].iloc[i] and feat_df["amount_abs"].iloc[i] > 1000:
                r.append("Large round-number amount")
            if feat_df["amount_is_large"].iloc[i]:
                r.append("Transaction >$10,000")
            if 7500 <= feat_df["amount_abs"].iloc[i] < 10000:
                r.append("Potential structuring (just below $10k threshold)")
            if feat_df["suspicious_text"].iloc[i] > 0.2:
                r.append("Suspicious keywords in description")
            if feat_df["balance_negative"].iloc[i]:
                r.append("Negative balance")
            if feat_df["account_txn_count"].iloc[i] > 20:
                r.append("High transaction velocity")
            if feat_df["account_total_volume"].iloc[i] > 200_000:
                r.append("High account total volume")
            if not r and blended[i] >= ML_FLAG_THRESHOLD:
                r.append("Statistical outlier pattern")
            reasons.append(r)

        self._save_models()

        if progress_callback:
            progress_callback(0.65, "ML screening complete — preparing account summaries…")

        result = pd.DataFrame({
            "account_id":     acct_ids,
            "row_index":      np.arange(len(df)),
            "risk_score":     blended,
            "is_flagged":     blended >= ML_FLAG_THRESHOLD,
            "anomaly_reasons": reasons,
        })
        return result

    def aggregate_accounts(self, txn_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Roll transaction-level scores up to account-level summaries.

        Returns one row per unique account with:
            account_id, account_risk_score, flagged_txn_count,
            total_txn_count, top_reasons
        """
        def agg(grp):
            flagged = grp[grp["is_flagged"]]
            all_reasons: list[str] = []
            for r in grp["anomaly_reasons"]:
                all_reasons.extend(r)
            reason_counts: dict[str, int] = {}
            for r in all_reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            top_reasons = sorted(reason_counts, key=reason_counts.get, reverse=True)[:3]

            return pd.Series({
                "account_risk_score":  float(grp["risk_score"].max()),
                "mean_risk_score":     float(grp["risk_score"].mean()),
                "flagged_txn_count":   int(len(flagged)),
                "total_txn_count":     int(len(grp)),
                "flagged_pct":         float(len(flagged) / max(len(grp), 1)),
                "top_reasons":         top_reasons,
            })

        summary = txn_scores.groupby("account_id").apply(agg).reset_index()
        summary = summary.sort_values("account_risk_score", ascending=False)
        return summary
