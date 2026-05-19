"""
Synthetic Fraud Dataset Generator for QLoRA Fine-Tuning.

Generates 1000 labeled account analysis examples covering:
  - Classic AML typologies (structuring, smurfing, layering, dormant, mule)
  - 2024-2025 emerging trends sourced from FINTRAC / FinCEN / FATF intelligence:
      * Pig-butchering / romance-investment scam (FATF 2024 Virtual Assets)
      * Elder financial abuse (FinCEN 2024-2025 priority)
      * Fentanyl funnel accounts (FinCEN FIN-2024-A002)
      * Crypto stablecoin layering (FATF stablecoin typology)
      * Authorized Push Payment / BEC deepfake fraud (FinCEN FIN-2024-Alert004)
      * Trade-Based Money Laundering / invoice manipulation (FATF TBML 2024)
      * AI / deepfake synthetic identity fraud (FinCEN FIN-2024-Alert004)
      * Cryptocurrency ATM structuring (FinCEN CVC kiosk guidance 2024)
  - Hard negatives (legitimate accounts flagged by ML — teach the model not to over-flag)

Usage:
    python training/generate_dataset.py
    # Outputs: training/fraud_dataset.jsonl (train) + training/fraud_dataset_eval.jsonl (eval)
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

random.seed(42)

OUTPUT_DIR   = Path(__file__).parent
TRAIN_FILE   = OUTPUT_DIR / "fraud_dataset.jsonl"
EVAL_FILE    = OUTPUT_DIR / "fraud_dataset_eval.jsonl"
EVAL_SPLIT   = 0.15   # 15% held out for evaluation

SYSTEM_PROMPT = """You are an expert AML (Anti-Money Laundering) analyst specializing in \
transaction fraud detection for fintech platforms. You analyze account activity and provide \
structured risk assessments.

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
- Off-hours (22:00–06:00 local): elevated risk indicator"""


# ── Helper functions ──────────────────────────────────────────────────────────────

def _acct() -> str:
    return f"ACC-{random.randint(10000, 99999)}"

def _ri(lo: int, hi: int) -> int:
    return random.randint(lo, hi)

def _rf(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


# ── Example generators ────────────────────────────────────────────────────────────

def _structuring_critical() -> tuple[str, str]:
    n_txns   = _ri(8, 15)
    amounts  = [_ri(8000, 9800) for _ in range(n_txns)]
    total    = sum(amounts)
    days     = _ri(14, 35)
    acct     = _acct()
    conf     = _ri(88, 97)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(80, 96)}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: {n_txns} cash deposits ranging ${min(amounts):,}–${max(amounts):,} "
        f"over {days} days, all just below $10,000 FINTRAC threshold."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: {n_txns} deposits clustered between ${min(amounts):,} and ${max(amounts):,} "
        f"over {days} days is a textbook structuring pattern. Each transaction is deliberately kept "
        f"below the $10,000 FINTRAC reporting threshold; the total structured amount (${total:,} CAD) "
        f"confirms intent to evade mandatory cash transaction reporting.\n"
        f"STR_NARRATIVE: Account {acct} conducted {n_txns} cash deposits ranging "
        f"${min(amounts):,}–${max(amounts):,} CAD over {days} days (total ${total:,} CAD), "
        f"each transaction deliberately kept below the $10,000 FINTRAC reporting threshold, "
        f"consistent with structuring of proceeds of crime."
    )
    return prompt, response


def _structuring_medium() -> tuple[str, str]:
    n_txns  = _ri(2, 4)
    amounts = [_ri(8500, 9700) for _ in range(n_txns)]
    total   = sum(amounts)
    acct    = _acct()
    conf    = _ri(55, 68)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(58, 72)}% | {n_txns + _ri(5, 15)} transactions | "
        f"Total volume: ${total + _ri(5000, 20000):,} CAD | "
        f"Anomalies: {n_txns} deposits in ${min(amounts):,}–${max(amounts):,} range. "
        f"Account has prior payroll history and normal spending patterns."
    )
    response = (
        f"VERDICT: MEDIUM\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: MONITOR\n"
        f"ANALYSIS: {n_txns} deposits near the $10,000 threshold is a weak structuring "
        f"signal, but the prior payroll history and normal spending provide plausible "
        f"alternative explanations. The small sample size prevents a definitive finding. "
        f"Recommend 30-day monitoring for pattern continuation or additional sub-threshold deposits.\n"
        f"STR_NARRATIVE: N/A"
    )
    return prompt, response


def _smurfing_critical() -> tuple[str, str]:
    n_txns  = _ri(25, 60)
    days    = _ri(3, 7)
    each    = _ri(800, 2500)
    total   = n_txns * each
    n_src   = _ri(5, 15)
    acct    = _acct()
    dest    = _acct()
    conf    = _ri(83, 95)
    velocity = round(n_txns / days, 1)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(75, 92)}% | {n_txns} transactions over {days} days | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: {n_txns} incoming deposits of ${each-200:,}–${each+300:,} each from "
        f"{n_src} different source accounts, all forwarded to {dest}. "
        f"Velocity: {velocity} txns/day."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: {n_txns} near-identical deposits from {n_src} distinct sources aggregating "
        f"${total:,} CAD over {days} days, all forwarded to a single destination, is a definitive "
        f"smurfing pattern. The aggregate value (${total:,}) would trigger mandatory reporting if "
        f"transferred as a single transaction. Velocity of {velocity} txns/day indicates coordinated execution.\n"
        f"STR_NARRATIVE: Account {acct} received {n_txns} structured deposits from {n_src} source "
        f"accounts aggregating ${total:,} CAD over {days} days, immediately forwarding proceeds to "
        f"account {dest}, consistent with coordinated smurfing to circumvent FINTRAC reporting thresholds."
    )
    return prompt, response


def _layering_critical() -> tuple[str, str]:
    n_txns        = _ri(15, 35)
    total         = _ri(150_000, 400_000)
    passthrough   = _ri(92, 99)
    n_dest        = _ri(2, 5)
    off_hrs_pct   = _ri(65, 90)
    acct          = _acct()
    conf          = _ri(86, 96)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(78, 94)}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Account receives large deposits and immediately re-transfers "
        f"{passthrough}% to {n_dest} different accounts within hours. "
        f"{off_hrs_pct}% of transactions occur between 01:00–04:00 EST. "
        f"Net account balance consistently near zero."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: A {passthrough}% pass-through rate with near-zero balance retention "
        f"to {n_dest} downstream accounts is a definitive layering indicator. "
        f"The {off_hrs_pct}% off-hours transaction concentration (01:00–04:00 EST) suggests "
        f"automated or coordinated execution, and the account functions purely as a transit node "
        f"to obscure the origin of ${total:,} CAD.\n"
        f"STR_NARRATIVE: Account {acct} received ${total:,} CAD and re-transferred {passthrough}% "
        f"to {n_dest} downstream accounts within hours, predominantly between 01:00–04:00 EST, "
        f"with near-zero balance retention, consistent with layering of proceeds of crime through "
        f"a transit account."
    )
    return prompt, response


def _dormant_high() -> tuple[str, str]:
    inactive_months = _ri(6, 24)
    total    = _ri(50_000, 200_000)
    n_txns   = _ri(4, 10)
    hours    = _ri(48, 96)
    baseline = _ri(200, 1500)
    mult     = round(total / baseline)
    acct     = _acct()
    conf     = _ri(78, 90)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(72, 90)}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Account dormant for {inactive_months} months (prior avg monthly volume: "
        f"${baseline:,}). Received ${total:,} across {n_txns} deposits in {hours} hours."
    )
    response = (
        f"VERDICT: HIGH\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: ESCALATE\n"
        f"ANALYSIS: A {inactive_months}-month dormant account receiving ${total:,} in {hours} "
        f"hours represents a {mult}× deviation from historical baseline — a strong dormant "
        f"activation signal. This pattern is frequently used to introduce externally accumulated "
        f"criminal proceeds through a low-scrutiny account. Source-of-funds verification is required.\n"
        f"STR_NARRATIVE: Account {acct}, dormant {inactive_months} months with historical monthly "
        f"activity of ${baseline:,}, received ${total:,} CAD in {n_txns} transactions over {hours} "
        f"hours — a {mult}× deviation from baseline consistent with dormant account activation for "
        f"introduction of proceeds of crime."
    )
    return prompt, response


def _mule_critical() -> tuple[str, str]:
    n_src   = _ri(4, 12)
    n_dest  = _ri(2, 4)
    total   = _ri(80_000, 300_000)
    n_txns  = _ri(20, 45)
    acct    = _acct()
    conf    = _ri(90, 97)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(82, 96)}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Receives deposits from {n_src} different source accounts, immediately "
        f"forwards to {n_dest} destination accounts within 12–36h. Net balance always returns "
        f"to near zero. No payroll, no utility payments, no POS transactions."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: Consistent receive-aggregate-forward behaviour from {n_src} sources to "
        f"{n_dest} destinations with near-zero balance retention and complete absence of "
        f"legitimate account activity (no payroll, utilities, or retail) eliminates all "
        f"lifestyle-spending explanations. This account is functioning as a money mule, "
        f"aggregating ${total:,} CAD from fragmented sources for coordinated forwarding.\n"
        f"STR_NARRATIVE: Account {acct} aggregated ${total:,} CAD from {n_src} source accounts "
        f"and forwarded proceeds to {n_dest} destination accounts within 12–36 hours per cycle, "
        f"maintaining near-zero balance with no legitimate activity, consistent with money mule "
        f"operations for proceeds-of-crime laundering."
    )
    return prompt, response


def _clear_legitimate() -> tuple[str, str]:
    n_txns    = _ri(30, 120)
    monthly   = _ri(3000, 8000)
    payroll   = _ri(2000, 5000)
    acct      = _acct()
    conf      = _ri(80, 94)
    ml_score  = _ri(38, 58)

    prompt = (
        f"Account {acct} | ML Risk: {ml_score}% | {n_txns} transactions | "
        f"Total volume: ${monthly * 3:,} CAD | "
        f"Anomalies: Slightly elevated ML score from transaction count normalization. "
        f"Account shows biweekly payroll of ${payroll:,}, regular utility bills, grocery "
        f"and gas POS transactions, mortgage payment. All transactions have clear merchant names."
    )
    response = (
        f"VERDICT: LOW\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: CLEAR\n"
        f"ANALYSIS: The ML flag is a false positive driven by transaction volume normalization. "
        f"The transaction pattern is entirely consistent with normal household activity — "
        f"regular payroll in, known merchant POS transactions, utility and mortgage payments out. "
        f"No structuring, smurfing, layering, or off-hours patterns present.\n"
        f"STR_NARRATIVE: N/A"
    )
    return prompt, response


def _velocity_false_positive() -> tuple[str, str]:
    n_txns   = _ri(60, 150)
    total    = _ri(15_000, 45_000)
    acct     = _acct()
    conf     = _ri(82, 93)
    ml_score = _ri(58, 72)

    prompt = (
        f"Account {acct} | ML Risk: {ml_score}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: High velocity ({round(n_txns/30, 1)} txns/day). However, all amounts are "
        f"$20–$950, diverse merchant categories (coffee shops, transit, groceries, streaming, "
        f"restaurants). Regular payroll confirmed."
    )
    response = (
        f"VERDICT: LOW\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: CLEAR\n"
        f"ANALYSIS: High transaction velocity driven entirely by normal consumer spending — "
        f"small amounts across diverse legitimate merchant categories with confirmed payroll. "
        f"Velocity without concentration, round amounts, or suspicious counterparties is not "
        f"an AML indicator. The ML score elevation is a normalization artifact. No action warranted.\n"
        f"STR_NARRATIVE: N/A"
    )
    return prompt, response


def _multi_typology_critical() -> tuple[str, str]:
    n_struct = _ri(10, 20)
    n_txns   = _ri(35, 60)
    total    = _ri(200_000, 500_000)
    n_dest   = _ri(3, 6)
    acct     = _acct()
    conf     = _ri(94, 99)
    ml_score = _ri(88, 98)
    velocity = round(n_txns / _ri(3, 6), 1)

    prompt = (
        f"Account {acct} | ML Risk: {ml_score}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: {n_struct} structured deposits ($8,200–$9,800 each), immediate forwarding "
        f"to {n_dest} accounts within 2h, {_ri(70, 90)}% of txns between 01:00–04:00 EST, "
        f"velocity {velocity} txns/day, no legitimate credits."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: Four simultaneous AML typologies — structuring ({n_struct} sub-threshold "
        f"deposits), layering (immediate forwarding to {n_dest} accounts), off-hours concentration "
        f"(01:00–04:00 EST), and extreme velocity ({velocity} txns/day). This is a sophisticated "
        f"coordinated laundering operation. Immediate STR filing, account freeze, and law "
        f"enforcement notification recommended.\n"
        f"STR_NARRATIVE: Account {acct} conducted {n_struct} structured deposits (total ${total:,} "
        f"CAD) between 01:00–04:00 EST at {velocity} transactions/day, immediately forwarding "
        f"proceeds to {n_dest} accounts, consistent with a multi-typology organized money "
        f"laundering scheme."
    )
    return prompt, response


def _off_hours_medium() -> tuple[str, str]:
    n_txns  = _ri(5, 12)
    total   = _ri(8_000, 25_000)
    acct    = _acct()
    conf    = _ri(50, 65)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(52, 66)}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: {_ri(3, n_txns)} round-number transfers (${_ri(1000,3000):,}, "
        f"${_ri(1000,5000):,}) between 02:00–04:30 EST. Account has payroll and normal daytime activity."
    )
    response = (
        f"VERDICT: MEDIUM\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: MONITOR\n"
        f"ANALYSIS: Off-hours round-number transfers are soft AML signals, but the account has "
        f"clear legitimate history (payroll, normal daytime spending) providing alternative "
        f"explanations such as automated scheduled payments or foreign time zone interactions. "
        f"Monitor for pattern development — if off-hours activity increases or amounts rise "
        f"toward the $10,000 threshold, escalate.\n"
        f"STR_NARRATIVE: N/A"
    )
    return prompt, response


def _business_wire_medium() -> tuple[str, str]:
    amount  = _ri(30_000, 150_000)
    acct    = _acct()
    conf    = _ri(52, 65)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(60, 72)}% | 3 transactions | "
        f"Total volume: ${amount:,} CAD | "
        f"Anomalies: Single international wire of ${amount:,} from overseas institution. "
        f"Account holder is a registered import/export business with 2 years of legitimate history."
    )
    response = (
        f"VERDICT: MEDIUM\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: MONITOR\n"
        f"ANALYSIS: A ${amount:,} international wire to a registered business is plausibly a "
        f"legitimate trade payment, but the amount warrants review and the overseas institution "
        f"requires enhanced due diligence on its FATF risk rating. No structuring, velocity, or "
        f"layering patterns present. Request supporting trade documentation (invoice, bill of lading) "
        f"before clearing.\n"
        f"STR_NARRATIVE: N/A"
    )
    return prompt, response


def _suspicious_text_high() -> tuple[str, str]:
    keywords = random.choice([
        ["hawala ref", "informal transfer", "USDT settlement"],
        ["crypto exchange", "BTC swap", "unhosted wallet"],
        ["cash courier", "hand delivery", "no receipt"],
        ["shell co transfer", "nominee acct", "offshore routing"],
    ])
    total = _ri(40_000, 150_000)
    acct  = _acct()
    conf  = _ri(74, 86)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(68, 82)}% | {_ri(5, 12)} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Transaction descriptions contain '{keywords[0]}', '{keywords[1]}', "
        f"'{keywords[2]}'. Two transactions exceed $15,000."
    )
    response = (
        f"VERDICT: HIGH\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: ESCALATE\n"
        f"ANALYSIS: Explicit references to '{keywords[0]}' and '{keywords[1]}' in transaction "
        f"descriptions combined with transactions exceeding $15,000 represent a high-confidence "
        f"typology cluster. These terms indicate informal value transfer or digital asset layering — "
        f"known AML vulnerabilities. Enhanced due diligence and counterparty identification required.\n"
        f"STR_NARRATIVE: Account {acct} conducted transactions referencing {keywords[0]} and "
        f"{keywords[1]} totalling ${total:,} CAD, including transactions exceeding $15,000, "
        f"consistent with informal value transfer and potential layering."
    )
    return prompt, response


def _negative_balance_high() -> tuple[str, str]:
    cycles  = _ri(4, 10)
    total   = _ri(20_000, 60_000)
    acct    = _acct()
    conf    = _ri(70, 82)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(65, 78)}% | {cycles * 3} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Account goes negative {cycles} times, each time receiving exact top-up "
        f"from anonymous sources. Descriptions: 'peer lending', 'P2P loan', 'informal credit'."
    )
    response = (
        f"VERDICT: HIGH\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: ESCALATE\n"
        f"ANALYSIS: Repeated negative-balance-then-exact-top-up cycles {cycles} times from "
        f"anonymous 'informal credit' sources suggests unlicensed money lending operating outside "
        f"regulated channels. Informal lending constitutes a predicate offence under PCMLTFA. "
        f"Counterparty identification and source-of-funds documentation required.\n"
        f"STR_NARRATIVE: Account {acct} experienced {cycles} negative-balance cycles immediately "
        f"offset by anonymous informal credit deposits totalling ${total:,} CAD, consistent with "
        f"unlicensed lending activity and potential PCMLTFA predicate offences."
    )
    return prompt, response


# ══════════════════════════════════════════════════════════════════════════════
# NEW GENERATORS — sourced from FINTRAC/FinCEN/FATF 2024-2025 intelligence
# ══════════════════════════════════════════════════════════════════════════════

def _pig_butchering_critical() -> tuple[str, str]:
    """
    Pig-butchering / sha zhu pan romance-investment scam.
    Source: FATF 2024 Virtual Assets report; FinCEN FIN-2024-Alert004.
    Pattern: dormant/normal account suddenly wires large sums to crypto VASPs
             after establishing online romantic contact. Victim is "coached".
    """
    total    = _ri(30_000, 250_000)
    n_wires  = _ri(3, 12)
    vasp     = random.choice(["Binance", "Coinbase", "Kraken", "Crypto.com", "unregistered VASP"])
    days     = _ri(14, 60)
    acct     = _acct()
    conf     = _ri(84, 95)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(75, 92)}% | {n_wires + _ri(20, 60)} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Account with 3+ years normal consumer history (payroll, POS spending) "
        f"suddenly conducts {n_wires} outgoing wires totalling ${total:,} to {vasp} "
        f"over {days} days. Customer contacted branch citing 'investment opportunity met online'. "
        f"No prior crypto activity. Amounts escalate each wire."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: Sudden pivot from normal consumer activity to {n_wires} escalating wires "
        f"to a crypto VASP (${total:,} total) after reported online romantic contact is the "
        f"definitive pig-butchering (sha zhu pan) pattern — ranked by FATF as a top-priority "
        f"2024 typology. The victim is being groomed to transfer savings to a fraudulent "
        f"investment platform. Funds are irrecoverable once converted to crypto.\n"
        f"STR_NARRATIVE: Account {acct} transferred ${total:,} CAD in {n_wires} escalating "
        f"wires to {vasp} over {days} days following reported online romantic contact, with no "
        f"prior cryptocurrency activity, consistent with pig-butchering investment fraud "
        f"(FATF 2024 Virtual Assets typology)."
    )
    return prompt, response


def _elder_financial_abuse_high() -> tuple[str, str]:
    """
    Elder financial exploitation.
    Source: FinCEN elder abuse SAR priority; OCC guidance 2024.
    Pattern: senior account suddenly used by new 'caregiver' / 'friend';
             large ATM withdrawals, P2P transfers, CD liquidations.
    """
    age      = _ri(72, 88)
    total    = _ri(20_000, 120_000)
    n_atm    = _ri(10, 30)
    n_p2p    = _ri(5, 15)
    acct     = _acct()
    conf     = _ri(79, 91)
    new_name = random.choice(["new caregiver", "new 'friend'", "new power-of-attorney", "new romantic interest"])

    prompt = (
        f"Account {acct} | ML Risk: {_ri(68, 84)}% | {n_atm + n_p2p + _ri(5, 10)} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Estimated {age}-year-old customer. Account shows {n_atm} ATM withdrawals "
        f"($200-$500 each at non-branch ATMs, late evening) and {n_p2p} P2P app transfers "
        f"(Interac e-Transfer) to a {new_name} added to account 6 weeks ago. "
        f"Savings account partially liquidated. Prior history: monthly pension + minimal spending."
    )
    response = (
        f"VERDICT: HIGH\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: ESCALATE\n"
        f"ANALYSIS: A combination of {n_atm} late-evening ATM withdrawals, {n_p2p} P2P "
        f"transfers to a recently added {new_name}, and partial savings liquidation in a "
        f"senior customer account constitutes a high-confidence elder financial abuse pattern. "
        f"This aligns with FinCEN's 2024-2025 priority typology. Immediate welfare check and "
        f"Adult Protective Services referral recommended alongside enhanced account restrictions.\n"
        f"STR_NARRATIVE: Account {acct} (estimated age {age}) exhibited {n_atm} non-branch "
        f"ATM withdrawals and {n_p2p} e-Transfer payments to a {new_name} totalling "
        f"${total:,} CAD, with concurrent savings liquidation, consistent with elder financial "
        f"exploitation (FinCEN priority typology 2024)."
    )
    return prompt, response


def _fentanyl_funnel_account_critical() -> tuple[str, str]:
    """
    Fentanyl / drug trafficking funnel account (cash structuring).
    Source: FinCEN FIN-2024-A002 ($1.4B in identified fentanyl-linked transactions).
    Pattern: multi-branch cash deposits just below CTR threshold, rapid outflow
             to Mexico-linked accounts or MSBs; drug-related memo text.
    """
    n_deposits = _ri(10, 25)
    each       = _ri(7_500, 9_800)
    total      = n_deposits * each
    n_branches = _ri(3, 8)
    acct       = _acct()
    conf       = _ri(89, 97)
    dest       = random.choice(["MSB wire service", "Mexico-linked account", "cryptocurrency kiosk", "unregistered VASP"])

    prompt = (
        f"Account {acct} | ML Risk: {_ri(85, 97)}% | {n_deposits + _ri(2, 5)} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: {n_deposits} cash deposits of ${each - 200:,}–${each:,} each across "
        f"{n_branches} different branch locations. Memo fields contain 'blues', 'ills', "
        f"'pharma'. Rapid outflow to {dest}. Account holder has no verifiable employment."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: Multi-branch structuring ({n_deposits} sub-CTR deposits across {n_branches} "
        f"locations) combined with fentanyl-related euphemisms ('blues', 'ills') and rapid "
        f"outflow to {dest} matches FinCEN FIN-2024-A002 fentanyl trafficking funnel account "
        f"typology. The $1.4 billion identified by FinCEN in 2024 followed exactly this pattern. "
        f"Mandatory FINTRAC STR filing and law enforcement referral required.\n"
        f"STR_NARRATIVE: Account {acct} conducted {n_deposits} structured cash deposits "
        f"(${each-200:,}–${each:,} each, total ${total:,} CAD) across {n_branches} branch "
        f"locations with drug-related memo text, immediately outflowing to {dest}, consistent "
        f"with fentanyl proceeds laundering (FinCEN FIN-2024-A002 typology)."
    )
    return prompt, response


def _crypto_stablecoin_layering_critical() -> tuple[str, str]:
    """
    Crypto / stablecoin layering into traditional banking.
    Source: FATF 2024 Virtual Assets report; stablecoins = bulk of illicit on-chain activity.
    Pattern: account receives rapid fiat-to-crypto-to-fiat round-trips;
             multiple VASP withdrawals; references to mixing/tumbling services.
    """
    n_txns   = _ri(8, 20)
    total    = _ri(50_000, 300_000)
    n_vasps  = _ri(2, 5)
    acct     = _acct()
    conf     = _ri(82, 94)
    coin     = random.choice(["USDT", "USDC", "DAI", "stablecoin"])

    prompt = (
        f"Account {acct} | ML Risk: {_ri(78, 93)}% | {n_txns} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: Account receives large fiat deposits then immediately purchases {coin} "
        f"via {n_vasps} different VASPs. {coin} subsequently sold and proceeds returned "
        f"as fiat wires from different institutions. Transaction descriptions reference "
        f"'swap', 'bridge', 'mixer'. Cross-chain activity detected."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: Fiat-to-{coin}-to-fiat round-trips across {n_vasps} VASPs with "
        f"explicit mixer/bridge references is a 2024 FATF priority typology — stablecoins "
        f"now represent the majority of illicit on-chain transaction volume globally. "
        f"The multi-VASP routing and cross-chain activity are deliberate layering to "
        f"obscure the origin of ${total:,} CAD.\n"
        f"STR_NARRATIVE: Account {acct} conducted fiat-to-{coin}-to-fiat conversion cycles "
        f"across {n_vasps} virtual asset service providers totalling ${total:,} CAD, with "
        f"explicit references to mixing and bridging services, consistent with stablecoin "
        f"layering (FATF 2024 Virtual Assets typology)."
    )
    return prompt, response


def _app_bec_fraud_critical() -> tuple[str, str]:
    """
    Authorized Push Payment (APP) / Business Email Compromise (BEC).
    Source: FinCEN FIN-2024-Alert004; deepfake + AI-enhanced BEC up 30% in 2025.
    Pattern: business account suddenly wires large amounts to new payee;
             payee bank details changed last-minute; no prior relationship.
    """
    amount   = _ri(25_000, 500_000)
    acct     = _acct()
    conf     = _ri(80, 93)
    trigger  = random.choice([
        "vendor bank details changed via email 2 days prior",
        "CFO impersonation email requesting urgent wire",
        "AI-generated voice call from 'CEO' authorizing payment",
        "invoice from lookalike vendor domain",
    ])

    prompt = (
        f"Account {acct} | ML Risk: {_ri(72, 88)}% | {_ri(3, 8)} transactions | "
        f"Total volume: ${amount:,} CAD | "
        f"Anomalies: Business account wired ${amount:,} to a new payee with no prior "
        f"transaction history. Wire preceded by: {trigger}. "
        f"Payee account opened <30 days ago at a different institution. "
        f"Customer now reports possible fraud and requests recall."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: A ${amount:,} wire to a new payee (<30-day-old account) triggered by "
        f"{trigger} is a textbook authorized push payment / BEC fraud pattern — FinCEN "
        f"FIN-2024-Alert004 notes AI-generated deepfakes are now used in the majority of "
        f"high-value BEC incidents. The customer's subsequent fraud report confirms the "
        f"transaction was induced by social engineering. Immediate recall attempt and "
        f"STR filing required.\n"
        f"STR_NARRATIVE: Account {acct} authorized a ${amount:,} CAD wire to a newly "
        f"established payee account following {trigger}, consistent with authorized push "
        f"payment fraud facilitated by AI-enhanced social engineering (FinCEN FIN-2024-Alert004)."
    )
    return prompt, response


def _tbml_trade_laundering_high() -> tuple[str, str]:
    """
    Trade-Based Money Laundering (TBML) — invoice manipulation.
    Source: FATF TBML guidance 2024; FFIEC BSA/AML manual.
    Pattern: import/export account with mismatched invoice values,
             third-party payments, phantom shipments.
    """
    amount   = _ri(80_000, 600_000)
    n_inv    = _ri(3, 8)
    acct     = _acct()
    conf     = _ri(74, 87)
    goods    = random.choice(["electronics", "industrial equipment", "precious metals", "textiles", "pharmaceuticals"])
    country  = random.choice(["UAE", "China", "Panama", "Cayman Islands", "Malaysia"])

    prompt = (
        f"Account {acct} | ML Risk: {_ri(68, 84)}% | {n_inv + _ri(3, 8)} transactions | "
        f"Total volume: ${amount:,} CAD | "
        f"Anomalies: Import business receives {n_inv} payments from a {country}-based "
        f"third party (not listed as the buyer on shipping documents) for {goods} invoiced "
        f"at 3-4x market value. Bills of lading reference quantities inconsistent with "
        f"invoice amounts. Same goods appear on multiple invoices. No warehouse footprint."
    )
    response = (
        f"VERDICT: HIGH\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: ESCALATE\n"
        f"ANALYSIS: Over-invoiced {goods} ({n_inv} invoices at 3-4x market value), payments "
        f"from unrelated {country} third parties, and inconsistent shipping documentation are "
        f"the three hallmarks of FATF-identified trade-based money laundering. The lack of "
        f"warehouse footprint suggests phantom shipments. TBML is FATF's highest-priority "
        f"non-crypto typology for 2024-2025.\n"
        f"STR_NARRATIVE: Account {acct} received ${amount:,} CAD from {country}-based third "
        f"parties for over-invoiced {goods} shipments with inconsistent shipping documentation, "
        f"consistent with trade-based money laundering via invoice manipulation "
        f"(FATF TBML typology 2024)."
    )
    return prompt, response


def _deepfake_identity_fraud_high() -> tuple[str, str]:
    """
    AI/Deepfake synthetic identity fraud.
    Source: FinCEN FIN-2024-Alert004 (November 2024); GenAI fraud up significantly.
    Pattern: recently opened account using AI-generated documents;
             immediate high-value activity inconsistent with stated profile.
    """
    amount   = _ri(15_000, 80_000)
    acct     = _acct()
    conf     = _ri(77, 90)
    doc_type = random.choice(["passport", "driver's licence", "utility bill", "pay stub"])

    prompt = (
        f"Account {acct} | ML Risk: {_ri(70, 87)}% | {_ri(5, 15)} transactions | "
        f"Total volume: ${amount:,} CAD | "
        f"Anomalies: Account opened 18 days ago using {doc_type} flagged by ID-verification "
        f"vendor as 'AI-generated / deepfake probability: 94%'. Stated income: $35,000/year. "
        f"Account immediately received ${amount:,} in wire transfers and attempted "
        f"full withdrawal. Selfie verification failed liveness check."
    )
    response = (
        f"VERDICT: HIGH\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: A 94%-probability AI-generated {doc_type}, failed liveness check, and "
        f"immediate ${amount:,} inflow-then-withdrawal pattern on an 18-day-old account "
        f"matches FinCEN FIN-2024-Alert004 — GenAI/deepfake identity fraud is the fastest-growing "
        f"vector in 2024-2025 SAR filings. Account should be frozen pending identity verification. "
        f"The incoming wire source should also be investigated as a potential victim account.\n"
        f"STR_NARRATIVE: Account {acct}, opened 18 days prior using a deepfake {doc_type} "
        f"(94% AI-generation probability per ID vendor), received and attempted to withdraw "
        f"${amount:,} CAD, consistent with synthetic identity fraud facilitated by generative AI "
        f"(FinCEN FIN-2024-Alert004)."
    )
    return prompt, response


def _crypto_atm_structuring_critical() -> tuple[str, str]:
    """
    Cryptocurrency ATM (Bitcoin kiosk) structuring.
    Source: FinCEN 2024-2025 CVC kiosk guidance; rising SAR category.
    Pattern: multiple crypto ATM deposits just below reporting thresholds;
             same wallet address reused; no verifiable income source.
    """
    n_kiosk  = _ri(8, 20)
    each     = _ri(7_000, 9_700)
    total    = n_kiosk * each
    n_locs   = _ri(2, 5)
    acct     = _acct()
    conf     = _ri(86, 95)

    prompt = (
        f"Account {acct} | ML Risk: {_ri(82, 95)}% | {n_kiosk + _ri(3, 8)} transactions | "
        f"Total volume: ${total:,} CAD | "
        f"Anomalies: {n_kiosk} cash-to-crypto ATM purchases of ${each-300:,}–${each:,} each "
        f"at {n_locs} different kiosk locations. Same destination Bitcoin wallet reused "
        f"across all transactions. No employment or verifiable income on file. "
        f"All purchases within 5-day window."
    )
    response = (
        f"VERDICT: CRITICAL\n"
        f"CONFIDENCE: {conf}%\n"
        f"ACTION: FILE_STR\n"
        f"ANALYSIS: {n_kiosk} sub-CTR crypto ATM purchases across {n_locs} kiosk locations "
        f"to a single Bitcoin wallet in 5 days is structuring via cryptocurrency kiosk — "
        f"a 2024-2025 FinCEN priority alert category. The wallet reuse eliminates any "
        f"anonymity argument. Total structured amount (${total:,}) confirms deliberate "
        f"threshold evasion with no legitimate income to explain source of funds.\n"
        f"STR_NARRATIVE: Account {acct} conducted {n_kiosk} structured cash-to-crypto ATM "
        f"purchases (${each-300:,}–${each:,} each, total ${total:,} CAD) across {n_locs} "
        f"kiosk locations to a single Bitcoin wallet over 5 days, consistent with "
        f"cryptocurrency kiosk structuring (FinCEN CVC kiosk SAR guidance 2024)."
    )
    return prompt, response


# ── Generator registry ────────────────────────────────────────────────────────────

_GENERATORS = [
    # Original typologies
    (_structuring_critical,         12),
    (_structuring_medium,            8),
    (_smurfing_critical,            10),
    (_layering_critical,            10),
    (_dormant_high,                  8),
    (_mule_critical,                10),
    (_clear_legitimate,             14),   # More negatives for calibration
    (_velocity_false_positive,       8),
    (_multi_typology_critical,      10),
    (_off_hours_medium,              8),
    (_business_wire_medium,          7),
    (_suspicious_text_high,          8),
    (_negative_balance_high,         7),
    # NEW — web-sourced 2024-2025 typologies (FINTRAC / FinCEN / FATF)
    (_pig_butchering_critical,      12),   # FATF top-2024 typology
    (_elder_financial_abuse_high,    9),   # FinCEN priority
    (_fentanyl_funnel_account_critical, 10),  # FinCEN FIN-2024-A002
    (_crypto_stablecoin_layering_critical, 10),  # FATF stablecoin
    (_app_bec_fraud_critical,       10),   # FinCEN FIN-2024-Alert004
    (_tbml_trade_laundering_high,    9),   # FATF TBML 2024
    (_deepfake_identity_fraud_high,  9),   # FinCEN FIN-2024-Alert004 GenAI
    (_crypto_atm_structuring_critical, 9), # FinCEN CVC kiosk
]


def _format_example_phi35(prompt: str, response: str) -> dict:
    """Format as Phi-3.5-mini SFT example."""
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n"
        f"<|assistant|>\n{response}<|end|>"
    )
    return {"text": text, "prompt": prompt, "response": response}


def generate_dataset(n_samples: int = 520) -> list[dict]:
    """Generate n_samples examples with weighted sampling across typologies."""
    generators   = [fn for fn, w in _GENERATORS for _ in range(w)]
    total_weight = len(generators)

    examples = []
    while len(examples) < n_samples:
        gen = random.choice(generators)
        try:
            prompt, response = gen()
            examples.append(_format_example_phi35(prompt, response))
        except Exception as e:
            print(f"Warning: generator {gen.__name__} failed: {e}")
            continue

    random.shuffle(examples)
    return examples


def main():
    print("Generating synthetic fraud dataset...")
    examples = generate_dataset(n_samples=1000)

    n_eval  = max(1, int(len(examples) * EVAL_SPLIT))
    n_train = len(examples) - n_eval

    train_examples = examples[:n_train]
    eval_examples  = examples[n_train:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"[OK] Dataset generated:")
    print(f"   Train: {len(train_examples)} examples -> {TRAIN_FILE}")
    print(f"   Eval:  {len(eval_examples)} examples  -> {EVAL_FILE}")
    print()

    # Show typology distribution
    verdicts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for ex in examples:
        for v in verdicts:
            if f"VERDICT: {v}" in ex["text"]:
                verdicts[v] += 1
                break

    print("Verdict distribution:")
    for v, c in verdicts.items():
        bar = "#" * (c // 5)
        print(f"  {v:<10} {c:3d}  {bar}")


if __name__ == "__main__":
    main()
