from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from createTranscript import createTranscript

# huggingface_dataset_name = "knkarthick/dialogsum"

# # Load dataset
# dataset = load_dataset(huggingface_dataset_name)

# print(dataset)


model_name = 'philschmid/bart-large-cnn-samsum'

# Ensure CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load model and tokenizer
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,  # Use bfloat16 if your GPU supports it
    device_map="auto"  # Automatically map model layers to available devices
).to(device)  # Move the model to GPU

tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

# Example index
# index = 201

# dialogue = dataset['test'][index]['dialogue']
# summary = dataset['test'][index]['summary']

# prompt = f"""
# Summarize the following conversation.

# {dialogue}

# Summary:
# """

prompt = createTranscript(3, "video1.mp4")

# Tokenize input and move it to GPU
inputs = tokenizer(prompt, return_tensors='pt').to(device)

output = tokenizer.decode(
    original_model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=500,  # Generate up to 500 tokens
    no_repeat_ngram_size=3,  # Avoid repetition of 3-grams
    temperature=0.7,         # Control randomness (lower values are more deterministic)
    top_p=0.9,               # Use nucleus sampling for diverse outputs
    )[0], 
    skip_special_tokens=True
)

dash_line = '-' * 100
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')