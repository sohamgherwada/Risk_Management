import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_ID = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_ID = "models/phi35-fraud-lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

print("Loading base model on GPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

prompt = """<|im_start|>system
You are a fraud detection AI. Analyze the transaction and output a verdict.
Format your response as:
VERDICT: [Fraud/Legitimate]
REASONING: [Your explanation]<|im_end|>
<|im_start|>user
Account ID: 99812
Transaction Amount: $9,500.00
Transaction Type: Wire Transfer
Location: Unknown (IP masked via VPN)
Time: 03:14 AM
Account Age: 2 days
Recent Activity: 4 similar wire transfers just under the $10k reporting limit in the last 24 hours.<|im_end|>
<|im_start|>assistant
"""

print("\n--- Input Prompt ---")
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n--- Generating Response ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=True,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("\n--- Model Output ---")
print(response)
print("\nTest completed.")
