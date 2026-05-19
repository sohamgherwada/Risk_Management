"""
Fraud Few-Shot Library — dynamically selected gold-standard examples.

15 hand-curated, expert-level fraud analysis examples covering all 5 AML
typologies. Each example is a complete VERDICT/CONFIDENCE/ACTION/ANALYSIS/
STR_NARRATIVE response for a realistic account profile.

At inference time, select_examples() picks the top-3 most relevant examples
based on the account's detected anomaly reasons, injecting them into the
prompt so the LLM behaves like a trained fraud expert immediately — no
fine-tuning required as a baseline, and they complement fine-tuned weights.
"""
from __future__ import annotations

from typing import List, Tuple

# ── Typology constants ────────────────────────────────────────────────────────────
TYPOLOGY_STRUCTURING      = "structuring"
TYPOLOGY_SMURFING         = "smurfing"
TYPOLOGY_LAYERING         = "layering"
TYPOLOGY_DORMANT          = "dormant_activation"
TYPOLOGY_MULE             = "mule_account"
TYPOLOGY_VELOCITY         = "high_velocity"
TYPOLOGY_ROUND_AMOUNT     = "round_amounts"
TYPOLOGY_OFF_HOURS        = "off_hours"
TYPOLOGY_LARGE_TXN        = "large_transaction"
TYPOLOGY_SUSPICIOUS_TEXT  = "suspicious_keywords"


# ── Few-shot examples ──────────────────────────────────────────────────────────────
# Each entry: (typologies: list[str], account_sketch: str, model_response: str)
# account_sketch = brief account context (injected as the "user" turn)
# model_response = perfect analyst output (injected as the "assistant" turn)

_EXAMPLES: List[Tuple[List[str], str, str]] = [

    # ── 1. Classic structuring ─────────────────────────────────────────────────
    (
        [TYPOLOGY_STRUCTURING, TYPOLOGY_ROUND_AMOUNT],
        (
            "Account ACC-00441 | ML Risk: 81% | 12 transactions | "
            "Total volume: $108,400 CAD | "
            "Anomalies: 9 transactions in $8,000–$9,500 range, all cash deposits, "
            "all occurring within a 3-week period."
        ),
        (
            "VERDICT: CRITICAL\n"
            "CONFIDENCE: 92%\n"
            "ACTION: FILE_STR\n"
            "ANALYSIS: Nine deposits clustered between $8,000 and $9,500 over 21 days "
            "is a textbook structuring pattern — transactions are deliberately kept just "
            "below the $10,000 FINTRAC reporting threshold. The consistent cash-deposit "
            "method and tight clustering leave no plausible legitimate explanation.\n"
            "STR_NARRATIVE: Account ACC-00441 conducted nine cash deposits ranging from "
            "$8,000 to $9,500 CAD over a 21-day period (total $108,400), each transaction "
            "deliberately structured below the $10,000 FINTRAC reporting threshold, "
            "consistent with money laundering via structuring."
        ),
    ),

    # ── 2. Smurfing (many small, same destination) ────────────────────────────
    (
        [TYPOLOGY_SMURFING, TYPOLOGY_VELOCITY],
        (
            "Account ACC-01782 | ML Risk: 74% | 38 transactions in 4 days | "
            "Total volume: $42,700 CAD | "
            "Anomalies: 38 outgoing transfers of $1,000–$1,500 each, all to account "
            "ACC-09910, extremely high velocity (9.5 txns/day)."
        ),
        (
            "VERDICT: HIGH\n"
            "CONFIDENCE: 85%\n"
            "ACTION: FILE_STR\n"
            "ANALYSIS: Thirty-eight transfers of similar small amounts (avg $1,124) "
            "all directed to a single destination account in 4 days is a clear smurfing "
            "pattern — aggregated value ($42,700) would trigger reporting thresholds if "
            "sent as a single transaction. Velocity of 9.5 txns/day is 12× the account "
            "baseline.\n"
            "STR_NARRATIVE: Account ACC-01782 conducted 38 structured transfers of "
            "$1,000–$1,500 CAD each to account ACC-09910 over a 4-day window, "
            "aggregating $42,700 through smurfing to avoid FINTRAC reporting obligations."
        ),
    ),

    # ── 3. Layering — rapid inter-account transfers ───────────────────────────
    (
        [TYPOLOGY_LAYERING, TYPOLOGY_VELOCITY, TYPOLOGY_OFF_HOURS],
        (
            "Account ACC-03341 | ML Risk: 79% | 22 transactions | "
            "Total volume: $287,500 CAD | "
            "Anomalies: Account receives large deposits then immediately re-transfers "
            "95% to 3 different accounts within hours. 18 of 22 transactions occur "
            "between 02:00–05:00 EST."
        ),
        (
            "VERDICT: CRITICAL\n"
            "CONFIDENCE: 89%\n"
            "ACTION: FILE_STR\n"
            "ANALYSIS: The receive-and-forward pattern with >95% pass-through rate "
            "and near-zero account retention is a definitive layering indicator. "
            "Off-hours timing (02:00–05:00 EST) suggests automated or coordinated "
            "execution. The account functions purely as a transit node to obscure "
            "the money trail across three downstream accounts.\n"
            "STR_NARRATIVE: Account ACC-03341 received $287,500 CAD across 22 "
            "transactions and immediately re-transferred 95% to three downstream "
            "accounts between 02:00–05:00 EST, exhibiting layering behaviour "
            "consistent with proceeds of crime being moved through a transit account."
        ),
    ),

    # ── 4. Dormant account sudden activation ──────────────────────────────────
    (
        [TYPOLOGY_DORMANT, TYPOLOGY_LARGE_TXN],
        (
            "Account ACC-07720 | ML Risk: 76% | 6 transactions | "
            "Total volume: $94,200 CAD | "
            "Anomalies: Account was inactive for 14 months, then received $94,200 "
            "across 6 transactions in 72 hours. Previous average monthly volume < $800."
        ),
        (
            "VERDICT: HIGH\n"
            "CONFIDENCE: 82%\n"
            "ACTION: ESCALATE\n"
            "ANALYSIS: A 14-month dormant account receiving $94,200 in 72 hours "
            "represents a 118× deviation from historical baseline — one of the "
            "strongest dormant-activation signals. This pattern is frequently used "
            "to introduce externally accumulated criminal proceeds into the financial "
            "system through a low-scrutiny account.\n"
            "STR_NARRATIVE: Account ACC-07720, inactive for 14 months with historical "
            "monthly activity below $800, received $94,200 CAD in 6 transactions over "
            "72 hours — a 118× deviation from baseline consistent with dormant account "
            "activation for money laundering purposes."
        ),
    ),

    # ── 5. Mule account (receive and forward) ─────────────────────────────────
    (
        [TYPOLOGY_MULE, TYPOLOGY_VELOCITY],
        (
            "Account ACC-02219 | ML Risk: 83% | 30 transactions | "
            "Total volume: $156,000 CAD | "
            "Anomalies: Account receives deposits from 8 different accounts then "
            "forwards funds to 2 accounts within 24–48h. Net account balance "
            "always returns near zero. No payroll or recurring legitimate credits."
        ),
        (
            "VERDICT: CRITICAL\n"
            "CONFIDENCE: 94%\n"
            "ACTION: FILE_STR\n"
            "ANALYSIS: Consistent receive-aggregate-forward behaviour from 8 sources "
            "to 2 destinations with net-zero balance retention is a canonical mule "
            "account profile. The absence of any payroll, government benefits, or "
            "recurring legitimate credits eliminates lifestyle-spending explanations. "
            "This account is operationally functioning as a money mule.\n"
            "STR_NARRATIVE: Account ACC-02219 aggregated $156,000 CAD from 8 distinct "
            "source accounts and forwarded proceeds to 2 destination accounts within "
            "24–48 hours per cycle, maintaining near-zero balance retention, consistent "
            "with money mule activity for proceeds-of-crime laundering."
        ),
    ),

    # ── 6. Moderate structuring — MEDIUM risk ─────────────────────────────────
    (
        [TYPOLOGY_STRUCTURING],
        (
            "Account ACC-05530 | ML Risk: 63% | 4 transactions | "
            "Total volume: $35,600 CAD | "
            "Anomalies: 2 cash deposits of $9,200 and $9,400 within 5 days. "
            "Account has some prior legitimate payroll history."
        ),
        (
            "VERDICT: MEDIUM\n"
            "CONFIDENCE: 61%\n"
            "ACTION: MONITOR\n"
            "ANALYSIS: Two deposits just below the $10,000 reporting threshold within "
            "5 days is a weak structuring signal, but the prior payroll history and "
            "small sample (only 2 suspect deposits) prevent a definitive finding. "
            "The amounts could reflect legitimate cash-business revenues. Recommend "
            "30-day enhanced monitoring for pattern continuation.\n"
            "STR_NARRATIVE: N/A"
        ),
    ),

    # ── 7. Off-hours + round amounts — LOW/MEDIUM ─────────────────────────────
    (
        [TYPOLOGY_OFF_HOURS, TYPOLOGY_ROUND_AMOUNT],
        (
            "Account ACC-08812 | ML Risk: 55% | 8 transactions | "
            "Total volume: $22,000 CAD | "
            "Anomalies: 5 round-number transfers ($1,000, $2,000, $5,000) at 03:15, "
            "03:45, 04:20 EST. Account shows payroll deposits and normal spending history."
        ),
        (
            "VERDICT: MEDIUM\n"
            "CONFIDENCE: 52%\n"
            "ACTION: MONITOR\n"
            "ANALYSIS: Off-hours round-number transfers are a soft AML signal, but "
            "the account has clear legitimate history (payroll, normal spending) that "
            "provides an alternative explanation. The odd hours may indicate an "
            "automated scheduled transfer or foreign time zone interaction. Not "
            "sufficient for escalation without additional pattern development.\n"
            "STR_NARRATIVE: N/A"
        ),
    ),

    # ── 8. High velocity — LOW risk (business context) ────────────────────────
    (
        [TYPOLOGY_VELOCITY],
        (
            "Account ACC-11234 | ML Risk: 61% | 95 transactions | "
            "Total volume: $48,200 CAD | "
            "Anomalies: High velocity (3.8 txns/day), but amounts are all small "
            "($50–$800), diverse merchants (groceries, gas, restaurants, utilities). "
            "Regular biweekly payroll deposits confirmed."
        ),
        (
            "VERDICT: LOW\n"
            "CONFIDENCE: 78%\n"
            "ACTION: CLEAR\n"
            "ANALYSIS: High transaction count driven by normal consumer spending — "
            "diverse merchant categories, small amounts, and confirmed payroll are "
            "all consistent with an active individual account. Velocity alone without "
            "concentration, round amounts, or cash patterns does not constitute an "
            "AML signal. ML score is elevated by txn count normalization artifact.\n"
            "STR_NARRATIVE: N/A"
        ),
    ),

    # ── 9. Suspicious keywords ─────────────────────────────────────────────────
    (
        [TYPOLOGY_SUSPICIOUS_TEXT, TYPOLOGY_LARGE_TXN],
        (
            "Account ACC-06601 | ML Risk: 72% | 7 transactions | "
            "Total volume: $68,400 CAD | "
            "Anomalies: Transaction descriptions contain 'crypto exchange', "
            "'USDT', 'hawala ref', 'informal transfer'. Two transactions > $15,000."
        ),
        (
            "VERDICT: HIGH\n"
            "CONFIDENCE: 79%\n"
            "ACTION: ESCALATE\n"
            "ANALYSIS: The combination of explicit hawala reference terminology, "
            "crypto exchange activity, and transactions exceeding $15,000 represents "
            "a high-confidence typology cluster. Hawala references indicate informal "
            "value transfer system usage — a known AML vulnerability. The crypto "
            "exchange activity adds a second layering vector.\n"
            "STR_NARRATIVE: Account ACC-06601 conducted transactions referencing "
            "hawala transfers and cryptocurrency exchanges totalling $68,400 CAD, "
            "including two transactions exceeding $15,000, consistent with informal "
            "value transfer and potential layering through digital assets."
        ),
    ),

    # ── 10. Multi-typology CRITICAL ───────────────────────────────────────────
    (
        [TYPOLOGY_STRUCTURING, TYPOLOGY_OFF_HOURS, TYPOLOGY_VELOCITY, TYPOLOGY_LAYERING],
        (
            "Account ACC-00099 | ML Risk: 94% | 47 transactions | "
            "Total volume: $321,000 CAD | "
            "Anomalies: 21 structured deposits ($8,500–$9,800 each), 34 of 47 txns "
            "between 01:00–04:00 EST, immediate re-forwarding to 5 accounts, "
            "velocity 11.2 txns/day over 4.2 days."
        ),
        (
            "VERDICT: CRITICAL\n"
            "CONFIDENCE: 97%\n"
            "ACTION: FILE_STR\n"
            "ANALYSIS: This account exhibits four simultaneous AML typologies — "
            "structuring (21 sub-threshold deposits), off-hours concentration (72% "
            "of transactions 01:00–04:00), rapid layering (immediate forwarding to 5 "
            "accounts), and extreme velocity. The 94% ML score combined with manual "
            "review confirms this is a sophisticated laundering operation, not a "
            "false positive. Immediate STR filing and account freeze recommended.\n"
            "STR_NARRATIVE: Account ACC-00099 conducted 21 structured cash deposits "
            "ranging $8,500–$9,800 CAD (total $321,000) between 01:00–04:00 EST, "
            "immediately forwarding proceeds to five downstream accounts at 11.2 "
            "transactions/day — consistent with an organized money laundering scheme "
            "involving structuring, layering, and off-hours automated execution."
        ),
    ),

    # ── 11. Large isolated transactions — MEDIUM ──────────────────────────────
    (
        [TYPOLOGY_LARGE_TXN],
        (
            "Account ACC-04455 | ML Risk: 64% | 3 transactions | "
            "Total volume: $52,000 CAD | "
            "Anomalies: Single incoming wire of $50,000 from overseas institution. "
            "Account holder is registered business (import/export). Normal prior history."
        ),
        (
            "VERDICT: MEDIUM\n"
            "CONFIDENCE: 58%\n"
            "ACTION: MONITOR\n"
            "ANALYSIS: A $50,000 international wire to a registered import/export "
            "business is plausibly a legitimate trade payment, but the amount triggers "
            "review and the overseas origin warrants enhanced due diligence on the "
            "counterparty institution. No structuring, layering, or velocity anomalies "
            "present. Request supporting trade documentation before escalating.\n"
            "STR_NARRATIVE: N/A"
        ),
    ),

    # ── 12. Negative balance pattern ──────────────────────────────────────────
    (
        [TYPOLOGY_VELOCITY, TYPOLOGY_SUSPICIOUS_TEXT],
        (
            "Account ACC-09987 | ML Risk: 68% | 18 transactions | "
            "Total volume: $31,400 CAD | "
            "Anomalies: Account repeatedly goes negative then receives exact top-up "
            "amounts from anonymous sources. Description fields contain 'peer lending', "
            "'P2P loan', 'informal credit'."
        ),
        (
            "VERDICT: HIGH\n"
            "CONFIDENCE: 73%\n"
            "ACTION: ESCALATE\n"
            "ANALYSIS: Repeated negative-balance-then-exact-top-up cycles from "
            "anonymous sources suggest a loan-sharking or predatory lending arrangement "
            "operating outside regulated channels. The P2P and informal credit "
            "references in transaction descriptions are consistent with unlicensed "
            "money lending, which constitutes a predicate offence under PCMLTFA. "
            "Enhanced due diligence and counterparty identification required.\n"
            "STR_NARRATIVE: Account ACC-09987 exhibited repeated negative balance "
            "cycles immediately offset by anonymous 'informal credit' deposits "
            "totalling $31,400, consistent with unlicensed lending activity and "
            "potential predicate offences under PCMLTFA."
        ),
    ),

    # ── 13. Clear account — no action ─────────────────────────────────────────
    (
        [],
        (
            "Account ACC-13301 | ML Risk: 42% | 55 transactions | "
            "Total volume: $18,900 CAD | "
            "Anomalies: Slightly elevated ML score due to high txn count. "
            "All transactions are POS retail, confirmed payroll, utility bills, "
            "mortgage payments. No structuring, no off-hours, no suspicious keywords."
        ),
        (
            "VERDICT: LOW\n"
            "CONFIDENCE: 88%\n"
            "ACTION: CLEAR\n"
            "ANALYSIS: The ML flag is a false positive driven by transaction count "
            "normalization. The transaction pattern is entirely consistent with normal "
            "household spending — payroll in, mortgage/utilities/POS out. No fraud "
            "typologies present. Account can be cleared with no further action.\n"
            "STR_NARRATIVE: N/A"
        ),
    ),

    # ── 14. Round amounts + high volume ───────────────────────────────────────
    (
        [TYPOLOGY_ROUND_AMOUNT, TYPOLOGY_LARGE_TXN, TYPOLOGY_VELOCITY],
        (
            "Account ACC-07761 | ML Risk: 77% | 14 transactions | "
            "Total volume: $195,000 CAD | "
            "Anomalies: 14 outgoing transfers of exactly $10,000, $15,000, $20,000 "
            "each. All to same beneficiary. No incoming credits to explain the source. "
            "Account opened 45 days ago."
        ),
        (
            "VERDICT: CRITICAL\n"
            "CONFIDENCE: 88%\n"
            "ACTION: FILE_STR\n"
            "ANALYSIS: Fourteen perfectly round-number transfers totalling $195,000 "
            "to a single beneficiary from a 45-day-old account with no visible income "
            "source is a severe multi-signal pattern. New account + large round amounts "
            "+ single beneficiary + no credits = high-probability layering or proceeds "
            "of crime disbursement. The account origin and source of funds must be "
            "investigated immediately.\n"
            "STR_NARRATIVE: Account ACC-07761, opened 45 days prior, conducted 14 "
            "round-number transfers ($10,000–$20,000 each, total $195,000 CAD) to a "
            "single beneficiary with no corresponding credit activity, consistent with "
            "disbursement of proceeds of crime through a recently opened transit account."
        ),
    ),

    # ── 15. Smurfing + dormant ────────────────────────────────────────────────
    (
        [TYPOLOGY_SMURFING, TYPOLOGY_DORMANT, TYPOLOGY_OFF_HOURS],
        (
            "Account ACC-02884 | ML Risk: 86% | 24 transactions | "
            "Total volume: $89,600 CAD | "
            "Anomalies: Account dormant for 8 months, then 24 incoming deposits "
            "of $2,000–$4,500 each from 12 different source accounts, all between "
            "23:00–02:00 EST. Immediate outgoing wire to offshore account."
        ),
        (
            "VERDICT: CRITICAL\n"
            "CONFIDENCE: 93%\n"
            "ACTION: FILE_STR\n"
            "ANALYSIS: Three high-confidence typologies converge: dormant account "
            "reactivation (8 months idle), smurfing (24 small deposits from 12 "
            "distinct sources aggregating $89,600), and off-hours execution "
            "(23:00–02:00). The immediate offshore outgoing wire adds a "
            "cross-border layering dimension. This is a coordinated laundering "
            "operation using a dormant account as a collection point.\n"
            "STR_NARRATIVE: Account ACC-02884, dormant for 8 months, received 24 "
            "deposits ranging $2,000–$4,500 from 12 source accounts between "
            "23:00–02:00 EST (total $89,600 CAD), immediately wired offshore, "
            "consistent with coordinated smurfing into a dormant collection account "
            "for cross-border money laundering."
        ),
    ),
]


# ── Keyword → typology mapping (for anomaly_reason → example selection) ───────────

_REASON_TO_TYPOLOGY: dict[str, str] = {
    "structuring":                          TYPOLOGY_STRUCTURING,
    "below $10k":                           TYPOLOGY_STRUCTURING,
    "threshold":                            TYPOLOGY_STRUCTURING,
    "round":                                TYPOLOGY_ROUND_AMOUNT,
    "velocity":                             TYPOLOGY_VELOCITY,
    "off-hours":                            TYPOLOGY_OFF_HOURS,
    "off hours":                            TYPOLOGY_OFF_HOURS,
    "large":                                TYPOLOGY_LARGE_TXN,
    ">$10,000":                             TYPOLOGY_LARGE_TXN,
    "suspicious keywords":                  TYPOLOGY_SUSPICIOUS_TEXT,
    "suspicious text":                      TYPOLOGY_SUSPICIOUS_TEXT,
    "high account total volume":            TYPOLOGY_VELOCITY,
    "statistical outlier":                  TYPOLOGY_VELOCITY,
}


def _reasons_to_typologies(anomaly_reasons: list[str]) -> set[str]:
    """Map detector anomaly reason strings to typology constants."""
    typologies: set[str] = set()
    for reason in anomaly_reasons:
        r = reason.lower()
        for keyword, typology in _REASON_TO_TYPOLOGY.items():
            if keyword in r:
                typologies.add(typology)
    return typologies


def select_examples(
    anomaly_reasons: list[str],
    n: int = 3,
) -> list[tuple[str, str]]:
    """
    Select the top-n most relevant few-shot examples based on detected anomaly reasons.

    Args:
        anomaly_reasons: List of reason strings from the detector (e.g. "Off-hours transaction").
        n: Number of examples to return (default 3).

    Returns:
        List of (account_sketch, model_response) tuples for injection into the prompt.
    """
    target_typologies = _reasons_to_typologies(anomaly_reasons)

    # Score each example by overlap with target typologies
    scored: list[tuple[int, int, tuple[str, str]]] = []
    for idx, (typologies, sketch, response) in enumerate(_EXAMPLES):
        overlap = len(set(typologies) & target_typologies)
        scored.append((overlap, idx, (sketch, response)))

    # Sort by overlap descending, then by original index for stability
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Never return more examples than exist
    n = min(n, len(scored))

    # Always include at least one CLEAR example to reduce false positive rate
    selected = [pair for _, _, pair in scored[:n]]
    has_clear = any("ACTION: CLEAR" in resp for _, resp in selected)
    if not has_clear and n >= 2:
        clear_pair = _EXAMPLES[12][1], _EXAMPLES[12][2]   # Example 13 (CLEAR)
        selected[-1] = clear_pair   # Replace the lowest-scoring with CLEAR

    return selected


def format_few_shot_block(examples: list[tuple[str, str]], model_family: str = "phi35") -> str:
    """
    Format selected examples as a few-shot block in the correct chat template.

    Args:
        examples: List of (account_sketch, response) from select_examples().
        model_family: 'phi35' or 'qwen' — controls token format.

    Returns:
        Formatted string to prepend to the inference prompt.
    """
    if not examples:
        return ""

    parts = ["--- FEW-SHOT EXAMPLES (reference only, do not repeat) ---\n"]

    for sketch, response in examples:
        if model_family == "phi35":
            parts.append(
                f"<|user|>\n{sketch}<|end|>\n"
                f"<|assistant|>\n{response}<|end|>\n"
            )
        else:  # qwen ChatML
            parts.append(
                f"<|im_start|>user\n{sketch}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>\n"
            )

    parts.append("--- END FEW-SHOT EXAMPLES ---\n")
    return "\n".join(parts)
