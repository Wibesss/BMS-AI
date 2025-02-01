from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

print(dataset)

model_name = 'google/flan-t5-base'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
).to(device) 

tokenizer = AutoTokenizer.from_pretrained(model_name)

index = 203

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt').to(device)

output = tokenizer.decode(
    original_model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=300,
    min_length=100,
    no_repeat_ngram_size=3, 
    temperature=0.7,    
    top_p=0.9,      
    )[0], 
    skip_special_tokens=True
)

dash_line = '-' * 100
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')


def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)

output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=4, 
    weight_decay=0.01,
    logging_steps=10, 
    per_device_train_batch_size=6,  
    save_steps=100,  
    save_strategy='steps',  
    save_total_limit=3, 
    evaluation_strategy='steps',  
    eval_steps=100  
)


trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train()
trainer.save_model("./BMS1") 

# from peft import LoraConfig, get_peft_model, TaskType

# lora_config = LoraConfig(
#     r=32, # Rank
#     lora_alpha=32,
#     target_modules=["q", "v"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
# )

# peft_model = get_peft_model(original_model, lora_config)

# output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

# peft_training_args = TrainingArguments(
#     output_dir=output_dir,
#     auto_find_batch_size=True,
#     learning_rate=1e-3,
#     num_train_epochs=4,
#     logging_steps=10,
#     per_device_train_batch_size=6,
#     save_strategy="steps",
#     save_steps=100,
#     save_total_limit=3
# )

    
# peft_trainer = Trainer(
#     model=peft_model,
#     args=peft_training_args,
#     train_dataset=tokenized_datasets["train"],
# )

# peft_trainer.train()

# peft_model_path="./peft-dialogue-summary-checkpoint-local"

# peft_trainer.model.save_pretrained(peft_model_path)
# tokenizer.save_pretrained(peft_model_path)