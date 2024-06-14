import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import load_dataset
from accelerate import Accelerator

peft_config = LoraConfig(
    r=4, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    task_type="CAUSAL_LM", 
    inference_mode=False, 
    # target_modules=["q_proj", "v_proj"],
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "c_proj"]    
)

model_id = "bigcode/starcoder2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map={"": Accelerator().process_index}, torch_dtype=torch.bfloat16)

ds = load_dataset("vdaita/editpackftmulti_inst")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def generate_text(row):
    patch = row["patch"]
    inst = row["inst"]
    file = row["old_contents"]
    filename = row["old_file"]
    row["text"] = f"""# Filename: {filename}\n# File:\n{file}\n# Instructions:\n{inst}\n# Patch:\n```diff\n{patch}\n```"""
    return row

ds = ds.map(generate_text, num_proc=20)
ds = ds.map(lambda samples: tokenizer(samples['text']), batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="finetuned_starcoder2",
    warmup_steps=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    optim="adafactor",
    torch_compile=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("finetuned_starcoder2")