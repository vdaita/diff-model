from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-7b")
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")

def formatting_prompts_func(row):
    patches = row["patch"]
    insts = row["inst"]
    files = row["old_contents"]
    filenames = row["old_file"]
    output_texts = []
    for i in range(len(patches)):
        output_texts.append(f"""# Filename: {filenames[i]}\n# File:\n{files[i]}\n# Instructions:\n{insts[i]}\n# Patch:\n```diff\n{patches[i]}\n```""")
    return output_texts

response_template = "# Patch:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="finetuned_starcoder2"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()