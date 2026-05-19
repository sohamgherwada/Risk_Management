"""
PDF Report Renderer — generates a printable PDF from the fraud analysis report.
Uses xhtml2pdf (pure Python, no native DLL deps) instead of WeasyPrint.
"""
from __future__ import annotations

import io
from datetime import datetime

from jinja2 import Template

_REPORT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; }
  body { font-family: Helvetica, Arial, sans-serif; color: #1e293b; font-size: 10pt; }

  /* ── Header ── */
  .header { background-color: #0f172a; color: #ffffff; padding: 20px 28px; }
  .header h1 { font-size: 18pt; font-weight: bold; }
  .header .meta { font-size: 8pt; color: #94a3b8; margin-top: 4px; }

  /* ── Badges ── */
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 8pt; font-weight: bold; }
  .badge-critical { background-color: #fef2f2; color: #991b1b; }
  .badge-high     { background-color: #fff7ed; color: #9a3412; }
  .badge-medium   { background-color: #fefce8; color: #854d0e; }
  .badge-low      { background-color: #f0fdf4; color: #166534; }

  /* ── Sections ── */
  .section { padding: 16px 28px; }
  .section h2 { font-size: 13pt; font-weight: bold; color: #0f172a;
                border-bottom: 1.5px solid #e2e8f0; padding-bottom: 5px; margin-bottom: 10px; }

  /* ── KPI table ── */
  .kpi-table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
  .kpi-table td { width: 33%; border: 1px solid #e2e8f0; border-radius: 6px;
                  padding: 12px; text-align: center; }
  .kpi-value { font-size: 20pt; font-weight: bold; color: #0f172a; }
  .kpi-value-fraud { font-size: 20pt; font-weight: bold; color: #dc2626; }
  .kpi-label { font-size: 7.5pt; color: #64748b; margin-top: 2px; }

  /* ── Account cards ── */
  .account-card { border: 1px solid #e2e8f0; border-radius: 6px;
                  padding: 12px; margin-bottom: 10px; }
  .account-id   { font-weight: bold; font-size: 11pt; }
  .account-meta { font-size: 8.5pt; color: #64748b; margin: 5px 0; }
  .analysis-text { font-size: 8.5pt; color: #334155; line-height: 1.5;
                   background-color: #f8fafc; padding: 7px; border-radius: 3px; margin-top: 5px; }
  .str-box { margin-top: 7px; background-color: #fef2f2; border: 1px solid #fca5a5;
             border-radius: 3px; padding: 7px; font-size: 8pt; color: #7f1d1d; }
  .reason-tag { background-color: #f1f5f9; border-radius: 3px; padding: 1px 6px;
                font-size: 7.5pt; color: #475569; margin-right: 3px; }

  /* ── Footer ── */
  .footer { padding: 14px 28px; background-color: #f8fafc;
            border-top: 1px solid #e2e8f0; font-size: 7.5pt; color: #94a3b8; }
</style>
</head>
<body>

<div class="header">
  <h1>Fraud Risk Analysis Report</h1>
  <div class="meta">
    File: {{ report.filename }} &nbsp;|&nbsp;
    Generated: {{ generated_at }} &nbsp;|&nbsp;
    Powered by Qwen2.5-7B + PolarQuant ONNX
  </div>
</div>

<div class="section">
  <h2>Executive Summary</h2>
  <table class="kpi-table">
    <tr>
      <td>
        <div class="kpi-value">{{ "{:,}".format(overview.total_transactions) }}</div>
        <div class="kpi-label">Total Transactions</div>
      </td>
      <td>
        <div class="kpi-value">{{ overview.total_accounts }}</div>
        <div class="kpi-label">Unique Accounts</div>
      </td>
      <td>
        <div class="kpi-value-fraud">{{ overview.fraud_percentage }}%</div>
        <div class="kpi-label">Flagged as Suspicious</div>
      </td>
    </tr>
    <tr>
      <td>
        <div class="kpi-value">{{ overview.high_risk_count }}</div>
        <div class="kpi-label">High / Critical Risk Accounts</div>
      </td>
      <td>
        <div class="kpi-value">{{ overview.str_recommended }}</div>
        <div class="kpi-label">STR Filing Recommended</div>
      </td>
      <td>
        <div class="kpi-value">{{ flagged_accounts|length }}</div>
        <div class="kpi-label">Accounts Analyzed by AI</div>
      </td>
    </tr>
  </table>
</div>

<div class="section">
  <h2>Flagged Account Details</h2>
  {% for acct in flagged_accounts %}
  <div class="account-card">
    <table width="100%"><tr>
      <td><span class="account-id">Account: {{ acct.account_id }}</span></td>
      <td align="right">
        <span class="badge badge-{{ acct.verdict|lower }}">
          {{ acct.verdict }} — {{ acct.confidence }}% confidence
        </span>
      </td>
    </tr></table>
    <div class="account-meta">
      ML Risk Score: {{ "%.0f"|format(acct.ml_risk_score * 100) }}% &nbsp;|&nbsp;
      Flagged Txns: {{ acct.flagged_txn_count }} / {{ acct.total_txn_count }} &nbsp;|&nbsp;
      Action: <strong>{{ acct.action }}</strong>
    </div>
    {% if acct.top_reasons %}
    <div style="margin: 4px 0;">
      {% for r in acct.top_reasons %}
      <span class="reason-tag">{{ r }}</span>
      {% endfor %}
    </div>
    {% endif %}
    <div class="analysis-text">{{ acct.analysis }}</div>
    {% if acct.str_narrative and acct.str_narrative != 'N/A' %}
    <div class="str-box"><strong>Recommended STR Narrative:</strong> {{ acct.str_narrative }}</div>
    {% endif %}
  </div>
  {% endfor %}
</div>

<div class="footer">
  <strong>CONFIDENTIAL — FOR AUTHORIZED COMPLIANCE USE ONLY</strong><br>
  This report is generated by an AI-assisted analysis system and must be reviewed by a qualified
  compliance officer before any regulatory action is taken. All STR filings require human authorization.
  System: NLP Risk Monitoring v1.0 | Model: Qwen2.5-7B INT4 (PolarQuant ONNX)
</div>

</body>
</html>
"""


def render_pdf(report: dict) -> bytes:
    """
    Render the report dict as a PDF using xhtml2pdf.
    Pure Python — no GTK/Pango/GLib native DLLs required.
    Returns raw PDF bytes.
    """
    try:
        from xhtml2pdf import pisa  # type: ignore
    except ImportError:
        raise RuntimeError("xhtml2pdf is not installed. Run: pip install xhtml2pdf")

    template = Template(_REPORT_HTML_TEMPLATE)
    html_str = template.render(
        report=report,
        overview=report.get("overview", {}),
        flagged_accounts=report.get("flagged_accounts", []),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    buf = io.BytesIO()
    pisa_status = pisa.CreatePDF(html_str, dest=buf)
    if pisa_status.err:
        raise RuntimeError(f"PDF generation failed (xhtml2pdf error code {pisa_status.err})")

    return buf.getvalue()
