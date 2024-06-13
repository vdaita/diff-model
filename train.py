import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import load_dataset

peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type="CAUSAL_LM", inference_mode=False)

model_id = "bigcode/starcoder2-15b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

dataset = load_dataset("vdaita/editpackftmulti_inst")

model = get_peft_model(model, peft_config)
model.print_trainable_params()

def tokenize_functions(examples):
    return tokenizer(examples["inst"], truncation=False)

training_args = TrainingArguments(
    output_dir="finetuned_starcoder2",
    learning_rate=1e-3,
    per_device_train_batch_size=32
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("finetuned_starcoder2")