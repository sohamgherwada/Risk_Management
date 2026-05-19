# Risk Management — Local Fraud Detection Engine

A locally-hosted fraud detection pipeline designed for fintech compliance teams. The system ingests transaction data, isolates suspicious accounts using a statistical machine learning ensemble, and routes flagged accounts to a local LLM (Qwen2.5-7B) for deep pattern analysis. It outputs a comprehensive risk report with a ranked account list, printable PDF exports, and actionable Suspicious Transaction Report (STR) narratives.

Zero cloud dependencies. 100% local execution.

---

## Model Setup

The system defaults to a **mock mode** if no local LLM is detected. In mock mode, the full pipeline operates normally (upload, ML screening, PDF generation, UI), but uses rule-based heuristics to generate the final narratives instead of LLM inference.

To deploy the full Qwen2.5-7B-INT4 model locally (~6GB VRAM required):

```powershell
.\venv\Scripts\Activate.ps1
python download_model.py
```

This pulls the ONNX INT4 model from Hugging Face into `models/qwen25-7b-int4/`. You only need to run this once.

---

## Pipeline Architecture

The analysis runs in three automated stages:

**Stage 1 — ML Screening**
An ensemble of IsolationForest and XGBoost evaluates every transaction. It flags statistical anomalies based on transaction amounts, off-hours execution, velocity spikes, and heuristic keyword matching. Accounts scoring below the `ML_FLAG_THRESHOLD` are cleared to prevent unnecessary LLM computation.

**Stage 2 — LLM Deep Analysis**
Flagged accounts are routed to Qwen2.5-7B. The model evaluates the full transaction history for each account, checking for known typologies (structuring, smurfing, layering, mule activity). It outputs a structured risk classification (CRITICAL / HIGH / MEDIUM / LOW), a confidence metric, and a draft FINTRAC STR narrative.

**Stage 3 — Reporting**
The pipeline generates a local HTML dashboard displaying the system-wide fraud rate, a ranked table of high-risk accounts, and the raw LLM analysis. Users can export the results to PDF or query the LLM regarding specific flagged accounts via the chat interface.

---

## PolarQuant KV Compression

PolarQuant is a custom quantization method integrated into the inference engine to reduce the VRAM footprint of the LLM's attention state (KV cache) during long-context processing.

Standard transformers store the KV cache in FP16. PolarQuant compresses this by converting Cartesian vectors to polar coordinates, quantizing the radius to INT8, and applying a Walsh-Hadamard transform before quantizing the angle to INT8.

**Benchmarks:**
- 2.67x KV cache compression
- 62.5% reduction in VRAM per inference call
- Near-lossless reconstruction (MSE ~0.000186)

PolarQuant is disabled by default to maximize generation speed. Enable it via `config.py` or environment variables if processing massive transaction histories that exceed your GPU's VRAM.

---

## Project Structure

```text
Risk_Management/
├── api/
│   ├── main.py                 # FastAPI application
│   └── routes/                 # Upload, report, and chat endpoints
├── core/
│   ├── detector.py             # ML ensemble (IsolationForest + XGBoost)
│   ├── feature_engineer.py     # Feature extraction logic
│   ├── schema_detector.py      # Automated column mapping
│   └── risk_queue.py           # In-memory job state tracking
├── dashboard/                  # Frontend UI (HTML/CSS/JS)
├── inference/
│   ├── chat_engine.py          # Post-report query handling
│   ├── llm_engine.py           # Qwen2.5-7B ONNX inference wrapper
│   ├── polar_quant.py          # KV cache compression logic
│   └── prompt_builder.py       # Context engineering
├── pdf/
│   └── report_renderer.py      # xhtml2pdf generation
└── config.py                   # Global configuration
```

---

## Installation & Execution

```powershell
# Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

*(If blocked by execution policies: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

**Start the server:**
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the dashboard at `http://127.0.0.1:8000`.

### Environment Configuration

Override default settings in `config.py` using environment variables:

```powershell
$env:POLARQUANT_ENABLED = "true"    # Compress KV cache for large context windows
$env:ML_FLAG_THRESHOLD = "0.60"     # Adjust ML screening strictness
$env:LLM_MAX_ACCOUNTS_PER_JOB = "10" # Cap LLM processing to top N accounts
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

| Variable | Default | Description |
|---|---|---|
| `ONNX_MODEL_PATH` | `models/qwen25-7b-int4` | Local path to the ONNX model |
| `POLARQUANT_ENABLED` | `false` | Toggle KV cache compression |
| `ML_FLAG_THRESHOLD` | `0.60` | Minimum ML score required for LLM analysis |
| `LLM_ESCALATE_THRESHOLD` | `0.75` | Minimum LLM score for HIGH/CRITICAL classification |
| `LLM_MAX_ACCOUNTS_PER_JOB` | `10` | Maximum accounts processed by the LLM per file |
| `MAX_UPLOAD_MB` | `500` | Maximum allowed file upload size |

---

## Compliance Note

The system automatically detects column schemas (amounts, IDs, timestamps) without requiring manual data formatting. All analysis runs entirely local; no transaction data is transmitted externally.

**Regulatory Disclaimer**: This tool generates draft narratives and preliminary risk assessments. A certified compliance officer must independently verify all findings before submitting formal Suspicious Transaction Reports (STRs) to FINTRAC or relevant regulatory bodies.
