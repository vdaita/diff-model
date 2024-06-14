import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import load_dataset
from accelerate import Accelerator

peft_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    task_type="CAUSAL_LM", 
    inference_mode=False, 
    target_modules=["q_proj", "v_proj"],
    bias="none"
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "c_proj"]    
)

model_id = "bigcode/starcoder2-15b"

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

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])

tokenized_ds = ds.map(preprocess_function, batched=True, num_proc=4, remove_columns=ds["train"].column_names)

block_size = 1024

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=4)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="finetuned_starcoder2",
    warmup_steps=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    per_device_train_batch_size=32,
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
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("finetuned_starcoder2")