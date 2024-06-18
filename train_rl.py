from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import ast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from torch import nn
from tqdm import tqdm
from diff_utils import *
from peft import LoraConfig, get_peft_model

def check_python(code): # https://stackoverflow.com/questions/4284313/python-how-to-check-syntax-of-python-file-script-without-executing-it
    try:
        ast.parse(code)
        return 1
    except SyntaxError:
        return -1

model_name = "./merged_model"

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map='auto',
    peft_config=peft_config
)

main_model_name = "bigcode/starcoder2-7b"

tokenizer = AutoTokenizer.from_pretrained(main_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Only finetune over python or javascript
ds = load_dataset("vdaita/editpackftmulti_inst")
ds = ds["train"]

def add_prompt_and_tokenize(row):
    row["input_text"] = f"""# Filename: {row['old_file']}\n# File:\n{row['old_contents']}\n# Instructions:\n{row['inst']}\n# Patch:\n```diff"""
    row["input_ids"] = tokenizer.encode(row["input_text"])
    return row

ds = ds.map(add_prompt_and_tokenize, num_proc=10)

trainer = PPOTrainer(
    model=model,
    config=PPOConfig(
        model_name="diff-starcoder",
        learning_rate=1.41e-5
    ),
    dataset=ds,
    tokenizer=tokenizer
)

print("Dataset: ", ds)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

epochs = 2

import re
def extract_code_block_data(md_text, language):
   # Regular expression pattern for matching diff code blocks
   pattern = rf'```{language}([\s\S]*?)```'
   code_blocks = re.findall(pattern, md_text, re.MULTILINE)
   return code_blocks

for epoch in tqdm(range(epochs), "epoch: "):
   for batch in tqdm(trainer.dataloader):
        query_tensors = batch["input_ids"]
        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        pairs = list(zip(batch["old_contents"], batch["response"]))
        rewards = []
        for (contents, patch_md) in pairs:
            diff_block = extract_code_block_data(patch_md)
            sr_blocks = parse_diff(diff_block)
            sr_blocks = [find_best_match(block.search_block, contents).block for block in sr_blocks]
            model_line_count = 0
            
            for block in sr_blocks:
                model_line_count += len(block.search_block.splitlines())
                model_line_count += len(block.replace_block.splitlines())
                contents = contents.replace(block.search_block, block.replace_block)
            
            truth_sr_blocks = parse_diff(batch["patch"])
            truth_line_count = 0
            for block in truth_sr_blocks:
                truth_line_count += len(block.search_block.splitlines())
                truth_line_count += len(block.replace_block.splitlines())

            rewards.append(torch.tensor(check_python(contents) * min(1, model_line_count/truth_line_count))) # This tuning tries to get the model to generate syntactically valid diffs. 

        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards)

trainer.save_pretrained("finetuned_starcoder2_rl")
