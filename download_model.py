"""
Download Phi-3.5-mini-instruct ONNX INT4 CUDA for the Risk Management system.

Why Phi-3.5-mini over Qwen2.5-7B?
  Your GPU is an RTX 5060 Ti with 8 GB VRAM.

  Model         Params  ONNX INT4 size  VRAM needed  Speed (RTX 5060 Ti)
  ─────────────────────────────────────────────────────────────────────────
  Phi-3.5-mini   3.8B       ~2.2 GB         ~3 GB     ✅ ~15–30 tok/s (fast)
  Qwen2.5-7B     7.0B       ~6.7 GB         ~7 GB     ⚠️  ~5–10 tok/s  (slow)
  Qwen2.5-32B   32.0B       ~18 GB          n/a       ❌ VRAM overflow

  Phi-3.5-mini is 3–5× faster and leaves ~5 GB free for the KV cache.
  It's from Microsoft's research team, trained on synthetic reasoning data,
  and follows strict output format instructions extremely well.

Usage:
    .\\venv\\Scripts\\Activate.ps1
    python download_model.py            # downloads Phi-3.5-mini (default)
    python download_model.py --model 7b # downloads Qwen2.5-7B instead

After downloading, the script prints the exact server start command.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub…")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).parent

MODELS = {
    "phi35": {
        "repo":   "microsoft/Phi-3.5-mini-instruct-onnx",
        # The CUDA INT4 weights live in a subfolder of this repo
        "subdir": "cuda/cuda-int4-rtn-block-32",
        "dest":   BASE_DIR / "models" / "phi35-mini-int4",
        "size":   "~2.2 GB",
        "vram":   "~3 GB",
        "family": "phi35",
        "layers": "32",
        "label":  "Phi-3.5-mini-instruct ONNX INT4 CUDA",
    },
    "7b": {
        "repo":   "keisuke-miyako/Qwen2.5-7B-onnx-int4",
        "subdir": None,
        "dest":   BASE_DIR / "models" / "qwen25-7b-int4",
        "size":   "~6.7 GB",
        "vram":   "~7 GB",
        "family": "qwen",
        "layers": "28",
        "label":  "Qwen2.5-7B ONNX INT4 CUDA",
    },
}

parser = argparse.ArgumentParser(description="Download ONNX model for Risk Management")
parser.add_argument(
    "--model",
    choices=list(MODELS.keys()),
    default="phi35",
    help="Which model to download (default: phi35)",
)
args = parser.parse_args()
cfg  = MODELS[args.model]

DEST: Path = cfg["dest"]
DEST.mkdir(parents=True, exist_ok=True)

print()
print("Model Download — Risk Management System")
print("=" * 55)
print(f"Model:       {cfg['label']}")
print(f"Source:      https://huggingface.co/{cfg['repo']}")
print(f"Destination: {DEST}")
print(f"Size:        {cfg['size']}")
print(f"VRAM needed: {cfg['vram']} (RTX 5060 Ti has 8 GB — fine)")
print()
print("Note: this model is separate from your EducAI project.")
print()

try:
    snapshot_download(
        repo_id          = cfg["repo"],
        local_dir        = str(DEST),
        allow_patterns   = [f"{cfg['subdir']}/**"] if cfg["subdir"] else None,
        ignore_patterns  = ["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )

    # If the model lives in a subdirectory, copy files up to root level
    if cfg["subdir"]:
        import shutil
        src = DEST / cfg["subdir"]
        if src.exists():
            for f in src.iterdir():
                target = DEST / f.name
                if not target.exists():
                    shutil.copy2(str(f), str(target))
            print(f"Copied weights from {cfg['subdir']}/ to model root.")

    print()
    print("✅ Download complete.")
    print()
    print("To start the server with this model:")
    print()
    print(f'  $env:ONNX_MODEL_PATH  = "{DEST}"')
    print(f'  $env:MODEL_FAMILY     = "{cfg["family"]}"')
    print(f'  $env:LLM_NUM_LAYERS   = "{cfg["layers"]}"')
    print( '  $env:POLARQUANT_ENABLED = "true"')
    print(f"  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload")

except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print()
    print("If you see a 401/authentication error:")
    print("  1. Create a free account at https://huggingface.co")
    print("  2. Run: huggingface-cli login")
    print("  3. Paste your token when prompted")
    print("  4. Run this script again")
    print()
    print("Or download manually from:")
    print(f"  https://huggingface.co/{cfg['repo']}/tree/main")
    print(f"  Place model files directly inside: {DEST}")
