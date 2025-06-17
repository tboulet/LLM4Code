from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
dtype_model = model.dtype

# Number of params
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

# Inference
prompt = "Q: What is the capital of France?\nA:"
tokens = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**tokens, max_length=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)

# Memory used
from memory import get_model_size, compute_memory
print(f"Memory used according to torch.cuda.memory_allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB\n")
print(f"Memory used according to the formula: {get_model_size(model_name, 'float32'):.2f} GB\n")
compute_memory(512, num_params, batch_size=1, embedding_size=768)

# dtype of model
dtype = model.dtype
print(f"Model dtype: {dtype}")