"""
Central configuration for the Risk Management NLP system.

Model families supported:
  'phi35'  — Microsoft Phi-3.5-mini-instruct ONNX INT4 (default, ~2.2 GB, fastest)
  'qwen'   — Qwen2.5-7B ONNX INT4 (~6.7 GB, more capable but slow on 8 GB VRAM)
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ─── ONNX / LLM ────────────────────────────────────────────────────────────────
# Default: Phi-3.5-mini-instruct ONNX INT4 CUDA (~2.2 GB, 3.8B params)
# 3–5× faster than Qwen2.5-7B on RTX 5060 Ti (8 GB VRAM)
# To use the larger 7B model: set ONNX_MODEL_PATH=models/qwen25-7b-int4
ONNX_MODEL_PATH = os.getenv(
    "ONNX_MODEL_PATH",
    str(BASE_DIR / "models" / "phi35-mini-int4")
)

# Model family — controls which chat template to use in prompt_builder.py
#   'phi35'  → <|user|>\n...<|end|>\n<|assistant|>\n
#   'qwen'   → <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
MODEL_FAMILY = os.getenv("MODEL_FAMILY", "phi35")

# Number of transformer attention layers in the loaded model
# Phi-3.5-mini = 32 layers, Qwen2.5-7B = 28 layers
LLM_NUM_LAYERS = int(os.getenv("LLM_NUM_LAYERS", "32"))

LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda:0")

# PolarQuant — KV cache compression analysis
# ENABLED   → runs compress/decompress on KV tensors and logs stats each generation
# MONITOR_ONLY → (always true with onnxruntime-genai) measures ratio but can't write back
#   Set POLARQUANT_MONITOR_ONLY=false only if you switch to a custom ONNX loop with KV write access
POLARQUANT_ENABLED      = os.getenv("POLARQUANT_ENABLED", "true").lower() == "true"
POLARQUANT_MONITOR_ONLY = os.getenv("POLARQUANT_MONITOR_ONLY", "true").lower() == "true"

# ─── RISK THRESHOLDS ───────────────────────────────────────────────────────────
ML_FLAG_THRESHOLD = float(os.getenv("ML_FLAG_THRESHOLD", "0.60"))   # score ≥ this goes to LLM
LLM_ESCALATE_THRESHOLD = float(os.getenv("LLM_ESCALATE_THRESHOLD", "0.75"))  # LLM score ≥ this = HIGH RISK

# ─── PROCESSING ────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── LLM BATCH ─────────────────────────────────────────────────────────────────
# Max accounts to send to LLM per job (protect against huge files)
LLM_MAX_ACCOUNTS_PER_JOB = int(os.getenv("LLM_MAX_ACCOUNTS_PER_JOB", "10"))

# ─── SERVER ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
