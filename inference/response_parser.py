"""
Response Parser — extracts structured data from QwQ-32B LLM output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMVerdict:
    verdict: str            # CRITICAL | HIGH | MEDIUM | LOW
    confidence: int         # 0–100
    action: str             # FILE_STR | ESCALATE | MONITOR | CLEAR
    analysis: str
    str_narrative: str
    raw_response: str

    @property
    def risk_level_color(self) -> str:
        return {
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#22c55e",
        }.get(self.verdict, "#94a3b8")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "action": self.action,
            "analysis": self.analysis,
            "str_narrative": self.str_narrative,
            "risk_level_color": self.risk_level_color,
        }


_DEFAULT_VERDICT = LLMVerdict(
    verdict="MEDIUM",
    confidence=50,
    action="MONITOR",
    analysis="Analysis could not be parsed from LLM response. Manual review recommended.",
    str_narrative="N/A",
    raw_response="",
)


def parse_response(raw: str) -> LLMVerdict:
    """
    Parse LLM structured output into an LLMVerdict.
    Handles minor formatting variations robustly.
    """
    lines = raw.strip().splitlines()

    verdict = _extract_field(lines, r"VERDICT:\s*(.+)")
    confidence_str = _extract_field(lines, r"CONFIDENCE:\s*(\d+)")
    action = _extract_field(lines, r"ACTION:\s*(.+)")
    analysis = _extract_field(lines, r"ANALYSIS:\s*(.+)")
    str_narrative = _extract_field(lines, r"STR_NARRATIVE:\s*(.+)")

    # Normalize verdict
    verdict = (verdict or "MEDIUM").upper().strip()
    if verdict not in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
        verdict = "MEDIUM"

    # Normalize action
    action = (action or "MONITOR").upper().strip()
    if action not in {"FILE_STR", "ESCALATE", "MONITOR", "CLEAR"}:
        action = "MONITOR"

    # Parse confidence
    try:
        confidence = max(0, min(100, int(confidence_str or "50")))
    except ValueError:
        confidence = 50

    return LLMVerdict(
        verdict=verdict,
        confidence=confidence,
        action=action,
        analysis=analysis or "See raw response.",
        str_narrative=str_narrative or "N/A",
        raw_response=raw,
    )


def _extract_field(lines: list[str], pattern: str) -> Optional[str]:
    """Find the first line matching `pattern` and return group 1."""
    for line in lines:
        m = re.search(pattern, line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None
