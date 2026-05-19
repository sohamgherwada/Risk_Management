"""
ONNX INT4 Export Script — converts fine-tuned Phi-3.5-mini to ONNX INT4 CUDA.

Takes the merged PyTorch weights from finetune_fraud.py and converts them to
the onnxruntime-genai format with CUDA INT4 quantization — the same format
the inference engine expects.

Prerequisites:
    pip install onnxruntime-genai-cuda olive-ai  (or use the official ONNX tools)

Usage:
    python training/export_to_onnx.py

Output:
    models/phi35-fraud-int4/        ← ready for the server
        model.onnx
        model.onnx.data
        genai_config.json
        tokenizer.json
        ...

After export, start the server with:
    $env:ONNX_MODEL_PATH = "models/phi35-fraud-int4"
    $env:MODEL_FAMILY    = "phi35"
    $env:LLM_NUM_LAYERS  = "32"
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
MERGED_DIR   = PROJECT_DIR / "models" / "phi35-fraud-merged"
ONNX_OUT_DIR = PROJECT_DIR / "models" / "phi35-fraud-int4"


def check_merged_weights():
    """Confirm merged PyTorch weights exist."""
    if not MERGED_DIR.exists() or not any(MERGED_DIR.glob("*.safetensors")):
        logger.error(f"Merged weights not found at {MERGED_DIR}")
        logger.error("Run first:  python training/finetune_fraud.py")
        sys.exit(1)
    logger.info(f"Found merged weights at: {MERGED_DIR}")


def try_olive_export():
    """
    Export using Microsoft Olive (recommended — same tool used for official builds).
    Olive handles quantization, graph optimization, and genai_config.json generation.
    """
    try:
        import olive  # type: ignore
    except ImportError:
        return False

    import json
    import tempfile

    logger.info("Exporting with Microsoft Olive (INT4 CUDA)…")
    ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Olive config for Phi-3.5-mini INT4 CUDA
    olive_config = {
        "input_model": {
            "type": "PyTorchModel",
            "model_path": str(MERGED_DIR),
            "model_type": "hf_model",
            "hf_config": {"model_name": "microsoft/Phi-3.5-mini-instruct", "task": "text-generation"},
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}],
            }
        },
        "data_configs": [],
        "passes": {
            "convert": {"type": "OnnxConversion", "target_opset": 17, "save_as_external_data": True},
            "quantize": {
                "type": "OnnxMatMul4Quantizer",
                "block_size": 32,
                "is_symmetric": True,
                "nodes_to_exclude": [],
            },
            "perf_tuning": {"type": "OrtPerfTuning", "enable_cuda_graph": True},
        },
        "output_dir": str(ONNX_OUT_DIR),
        "host": "local_system",
        "target": "local_system",
    }

    config_path = Path(tempfile.mktemp(suffix=".json"))
    config_path.write_text(json.dumps(olive_config, indent=2))

    result = subprocess.run(
        [sys.executable, "-m", "olive.workflows.run", "--config", str(config_path)],
        check=False,
    )
    config_path.unlink(missing_ok=True)

    if result.returncode == 0:
        logger.info(f"✅ Olive export successful → {ONNX_OUT_DIR}")
        _copy_tokenizer_files()
        return True

    logger.warning("Olive export failed — trying onnxruntime-genai builder…")
    return False


def try_ort_genai_export():
    """
    Export using onnxruntime-genai's built-in model builder.
    This is the easiest method if onnxruntime-genai is already installed.
    """
    try:
        import onnxruntime_genai.models.builder as builder  # type: ignore
        has_builder = True
    except ImportError:
        has_builder = False

    if not has_builder:
        # Fall back to CLI if the Python API isn't available
        result = subprocess.run(
            [sys.executable, "-m", "onnxruntime_genai.models.builder",
             "--model", str(MERGED_DIR),
             "--output", str(ONNX_OUT_DIR),
             "--precision", "int4",
             "--execution_provider", "cuda",
             "--quantization_method", "rtn"],
            check=False,
        )
        if result.returncode != 0:
            return False
    else:
        ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = PROJECT_DIR / "models" / "phi35-fraud-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        builder.create_model(
            model_name="microsoft/Phi-3.5-mini-instruct",
            input_path=str(MERGED_DIR),
            output_dir=str(ONNX_OUT_DIR),
            precision="int4",
            execution_provider="cuda",
            cache_dir=str(cache_dir)
        )

    logger.info(f"✅ onnxruntime-genai export successful → {ONNX_OUT_DIR}")
    return True


def _copy_tokenizer_files():
    """Copy tokenizer files from merged dir to ONNX output dir."""
    import shutil
    tok_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
    ]
    for fname in tok_files:
        src = MERGED_DIR / fname
        dst = ONNX_OUT_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))


def print_manual_instructions():
    """Print manual export instructions as fallback."""
    logger.warning("Automatic export tools not available.")
    print()
    print("=" * 65)
    print("MANUAL EXPORT INSTRUCTIONS")
    print("=" * 65)
    print()
    print("Option A — Microsoft Olive (recommended):")
    print("  pip install olive-ai[gpu]")
    print("  python training/export_to_onnx.py")
    print()
    print("Option B — onnxruntime-genai builder:")
    print("  pip install onnxruntime-genai-cuda")
    print("  python -m onnxruntime_genai.models.builder \\")
    print(f"    --model {MERGED_DIR} \\")
    print(f"    --output {ONNX_OUT_DIR} \\")
    print("    --precision int4 \\")
    print("    --execution_provider cuda \\")
    print("    --quantization_method rtn")
    print()
    print("Option C — Use the adapter directly (slower, PyTorch inference):")
    print("  pip install peft transformers accelerate bitsandbytes")
    print("  # Then set ONNX_MODEL_PATH to the merged dir and update")
    print("  # llm_engine.py to use transformers instead of onnxruntime-genai")
    print()
    print(f"Merged weights location: {MERGED_DIR}")
    print(f"Target ONNX output dir:  {ONNX_OUT_DIR}")
    print()


def print_server_launch_instructions():
    print()
    print("=" * 65)
    print("SERVER LAUNCH (after export)")
    print("=" * 65)
    print()
    print("  $env:ONNX_MODEL_PATH    = 'models/phi35-fraud-int4'")
    print("  $env:MODEL_FAMILY       = 'phi35'")
    print("  $env:LLM_NUM_LAYERS     = '32'")
    print("  $env:POLARQUANT_ENABLED = 'true'")
    print("  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload")
    print()


def main():
    check_merged_weights()
    ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try export methods in order of preference
    success = try_ort_genai_export() or try_olive_export()

    if not success:
        print_manual_instructions()
        return

    _copy_tokenizer_files()

    logger.info("")
    logger.info(f"✅ Fine-tuned ONNX model ready at: {ONNX_OUT_DIR}")
    print_server_launch_instructions()


if __name__ == "__main__":
    main()
