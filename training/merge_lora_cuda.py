import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge():
    base_id = "microsoft/Phi-3.5-mini-instruct"
    adapter_dir = "models/phi35-fraud-lora"
    merged_dir = "models/phi35-fraud-merged"
    
    print(f"Loading base model in bfloat16 on CUDA (GPU) from {base_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map="cuda"
    )
    
    print(f"Loading PEFT adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    print("Merging adapters into base model natively on GPU (high-precision)...")
    merged_model = peft_model.merge_and_unload()
    
    # Workaround for Transformers tied weights saving
    merged_model.tie_weights = lambda: None
    if hasattr(merged_model, '_tied_weights_keys'):
        merged_model._tied_weights_keys = []
    
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
        
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
    tokenizer.save_pretrained(merged_dir)
    print("✅ High-precision CUDA Merge complete!")

if __name__ == "__main__":
    merge()
