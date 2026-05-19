"""
QLoRA Fine-Tuning Script — Phi-3.5-mini-instruct → Fraud AML Expert.

This script fine-tunes the Phi-3.5-mini-instruct model (PyTorch weights,
NOT the ONNX version) using QLoRA (4-bit base weights + LoRA adapters),
then merges the adapters for ONNX export.

Hardware requirements:
  - RTX 5060 Ti (8 GB VRAM) ✅ — QLoRA of 3.8B model fits in ~5 GB
  - Training time: ~2–4 hours for 3 epochs on 520 examples
  - Disk space: ~8 GB (PyTorch weights) + ~300 MB (LoRA adapter)

Pipeline:
  Step 1: python training/generate_dataset.py      # ~5 sec
  Step 2: python training/finetune_fraud.py         # ~2-4 hours
  Step 3: python training/export_to_onnx.py         # ~30 min

Usage:
    # Activate venv first
    .\\venv\\Scripts\\Activate.ps1

    # Install training deps (one-time)
    pip install peft trl bitsandbytes datasets accelerate

    # Generate dataset (if not already done)
    python training/generate_dataset.py

    # Fine-tune
    python training/finetune_fraud.py

    # (Optional) Resume from checkpoint
    python training/finetune_fraud.py --resume

    # Use a different base model
    python training/finetune_fraud.py --base-model microsoft/Phi-3.5-mini-instruct
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
TRAIN_FILE   = SCRIPT_DIR / "fraud_dataset.jsonl"
EVAL_FILE    = SCRIPT_DIR / "fraud_dataset_eval.jsonl"
ADAPTER_DIR  = PROJECT_DIR / "models" / "phi35-fraud-lora"
MERGED_DIR   = PROJECT_DIR / "models" / "phi35-fraud-merged"

# ── Default hyperparameters ────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LORA_R             = 16      # LoRA rank — higher = more capacity, more VRAM
LORA_ALPHA         = 32      # LoRA alpha (typically 2×rank)
LORA_DROPOUT       = 0.05
TARGET_MODULES     = ["q_proj", "v_proj", "k_proj", "o_proj"]   # attention projections
LEARNING_RATE      = 2e-4
WARMUP_RATIO       = 0.03
NUM_EPOCHS         = 3
BATCH_SIZE         = 2       # per-device — small for 8 GB VRAM
GRAD_ACCUM         = 8       # effective batch = 2 × 8 = 16
MAX_SEQ_LEN        = 1024    # truncation length
SAVE_STEPS         = 50
EVAL_STEPS         = 50
LOGGING_STEPS      = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Phi-3.5-mini for fraud AML")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="HuggingFace model ID or local path for base model")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint in ADAPTER_DIR")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging LoRA weights after training")
    return parser.parse_args()


def check_prerequisites():
    """Verify all required packages and data files exist."""
    missing_pkgs = []
    for pkg in ["peft", "trl", "bitsandbytes", "datasets", "transformers", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)

    if missing_pkgs:
        raise ImportError(
            f"Missing packages: {', '.join(missing_pkgs)}\n"
            f"Install with: pip install {' '.join(missing_pkgs)}"
        )

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_FILE}\n"
            f"Run first: python training/generate_dataset.py"
        )

    logger.info("✅ Prerequisites satisfied")


def load_model_and_tokenizer(base_model: str):
    """Load base model in 4-bit (QLoRA) and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False   # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Phi-3.5 needs right padding for SFT

    logger.info(f"✅ Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, tokenizer


def apply_lora(model, lora_r: int = LORA_R):
    """Apply LoRA adapters to the model for efficient fine-tuning."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare for 4-bit training (enables gradient checkpointing, casts LN to fp32)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_dataset(tokenizer):
    """Load and tokenize the fraud dataset."""
    from datasets import load_dataset as hf_load

    data_files = {"train": str(TRAIN_FILE)}
    if EVAL_FILE.exists():
        data_files["validation"] = str(EVAL_FILE)

    dataset = hf_load("json", data_files=data_files)
    logger.info(f"Dataset: {len(dataset['train'])} train examples"
                + (f", {len(dataset['validation'])} eval" if "validation" in dataset else ""))
    return dataset


def train(args: argparse.Namespace):
    """Main training loop."""
    import torch
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments

    check_prerequisites()

    model, tokenizer = load_model_and_tokenizer(args.base_model)
    model            = apply_lora(model, lora_r=args.lora_r)
    dataset          = load_dataset(tokenizer)
    cols_to_remove = [c for c in ["prompt", "response"] if c in dataset["train"].column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # SFT config — note: SFTConfig extends TrainingArguments
    sft_config = SFTConfig(
        output_dir                  = str(ADAPTER_DIR),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = args.lr,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        fp16                        = False,
        bf16                        = True,
        optim                       = "paged_adamw_8bit",   # memory-efficient optimizer
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        eval_steps                  = EVAL_STEPS if "validation" in dataset else None,
        eval_strategy               = "steps" if "validation" in dataset else "no",
        load_best_model_at_end      = "validation" in dataset,
        max_length              = MAX_SEQ_LEN,
        dataset_text_field          = "text",
        packing                     = False,   # No packing — each example is a complete chat
        report_to                   = "none",  # Disable wandb/tensorboard unless you add them
    )

    trainer = SFTTrainer(
        model       = model,
        args        = sft_config,
        train_dataset = dataset["train"],
        eval_dataset  = dataset.get("validation"),
        processing_class = tokenizer,
    )

    logger.info("=" * 60)
    logger.info("Starting QLoRA fine-tuning")
    logger.info(f"  Base model:  {args.base_model}")
    logger.info(f"  LoRA rank:   {args.lora_r}")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  LR:          {args.lr}")
    logger.info(f"  Output dir:  {ADAPTER_DIR}")
    logger.info("=" * 60)

    resume_from = str(ADAPTER_DIR) if args.resume and ADAPTER_DIR.exists() else None
    trainer.train(resume_from_checkpoint=resume_from)

    # Save the LoRA adapter weights
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    logger.info(f"✅ LoRA adapter saved to: {ADAPTER_DIR}")

    if not args.no_merge:
        merge_and_save(args.base_model, tokenizer)


def merge_and_save(base_model: str, tokenizer=None):
    """Merge LoRA adapters into the base model and save full weights."""
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    logger.info("Merging LoRA adapters into base model…")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Load the base model in fp16 (no 4-bit — we need full weights for merging)
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged_model = merged_model.merge_and_unload()
    
    # Patch to prevent transformers 5.8.1 crash when saving tied weights
    merged_model.tie_weights = lambda: None
    if hasattr(merged_model, '_tied_weights_keys'):
        merged_model._tied_weights_keys = []
        
    merged_model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    logger.info(f"✅ Merged model saved to: {MERGED_DIR}")
    logger.info("")
    logger.info("Next step — export to ONNX INT4:")
    logger.info(f"  python training/export_to_onnx.py")


if __name__ == "__main__":
    args = parse_args()
    train(args)
