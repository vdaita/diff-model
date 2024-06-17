from datasets import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model_name = "bigcode/starcoder2-7b"
ds = load_dataset("vdaita/editpackftmulti_inst")

def generate_text(row):
    patch = row["patch"]
    inst = row["inst"]
    file = row["old_contents"]
    filename = row["old_file"]
    row["text"] = f"""# Filename: {filename}\n# File:\n{file}\n# Instructions:\n{inst}\n# Patch:\n```diff\n{patch}\n```"""
    return row

ds = ds.map(generate_text, num_proc=10)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

"""### Training"""

import transformers
from datasets import load_dataset

ds = ds.map(lambda samples: tokenizer(samples['text']), batched=True)

trainer = transformers.Trainer( # Maybe consider a custom trainer that only extracts the diffs themselves and leaves initial text alone?
    model=model,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='finetuned_starcoder2'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model("finetuned_starcoder2")
model.push_to_hub("vdaita/diff-starcoder-7b")