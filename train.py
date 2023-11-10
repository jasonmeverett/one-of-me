import json
import oneofme as me
import torch 


from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

model_checkpoint = "gpt2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

model_input = "What is the capital of France?"
inputs = tokenizer(model_input, return_tensors="pt")

print("Runnign sampling...")
model_out2 = model.sample(inputs.input_ids, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, temperature=0.7, top_p=0.92)

decoded = tokenizer.decode(model_out2[0])
print(decoded)
print(tokenizer.decode(tokenizer.eos_token_id))