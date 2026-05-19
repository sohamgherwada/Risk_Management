import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge():
    base_id = "microsoft/Phi-3.5-mini-instruct"
    adapter_dir = "models/phi35-fraud-lora"
    merged_dir = "models/phi35-fraud-merged"
    
    print(f"Loading base model from {base_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu"
    )
    
    print(f"Loading PEFT adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    print("Merging adapters into base model...")
    merged_model = peft_model.merge_and_unload()
    
    # Workaround for Transformers 5.8.1 tied weights saving bug
    # Just force the model to not look for tied weights during save
    merged_model.tie_weights = lambda: None
    if hasattr(merged_model, '_tied_weights_keys'):
        merged_model._tied_weights_keys = []
    
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {merged_dir}...")
    try:
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
    except Exception as e:
        print(f"Save failed with error: {e}. Attempting manual save...")
        # Fallback manual state_dict save if transformers is deeply bugged
        merged_model.config.save_pretrained(merged_dir)
        import safetensors.torch
        safetensors.torch.save_file(merged_model.state_dict(), f"{merged_dir}/model.safetensors")
        
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)
    print("✅ Merge complete!")

if __name__ == "__main__":
    merge()
