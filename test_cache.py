import torch
from transformers.cache_utils import DynamicCache

cache = DynamicCache()
print("Cache class:", type(cache))
print("Cache base classes:", type(cache).__mro__)
print("Initial seen_tokens:", getattr(cache, "seen_tokens", "N/A"))

# Simulate a prefill update
k = torch.zeros((1, 32, 14, 96))
v = torch.zeros((1, 32, 14, 96))
cache.update(k, v, 0)

print("After update 0, len(cache.layers):", len(cache.layers))
print("After update 0, get_seq_length(0):", cache.get_seq_length(0))
print("After update 0, seen_tokens:", getattr(cache, "seen_tokens", "N/A"))
