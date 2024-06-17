from trl import PPOConfig, PPOTrainer
import ast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from torch import nn
from tqdm import tqdm
from diff_utils import *

def check_python(code): # https://stackoverflow.com/questions/4284313/python-how-to-check-syntax-of-python-file-script-without-executing-it
    try:
        ast.parse(code)
        return 1
    except SyntaxError:
        return -1

model_name = "bigcode/starcoder2-7b"
peft_model_name = "vdaita/diff-starcoder-7b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map='auto',
)
model.load_adapter(peft_model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

"""### Post-processing on the model

Finally, we need to apply some post-processing on the 8-bit model to enable training, let's freeze all our layers, and cast the layer-norm in `float32` for stability. We also cast the output of the last layer in `float32` for the same reasons.
"""

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

# Only finetune over python or javascript
ds = load_dataset("vdaita/editpackft_inst")
ds = ds["train"]


trainer = PPOTrainer(
    model=model,
    config=PPOConfig(
        model_name="diff-starcoder",
        learning_rate=1.41e-5
    ),
    dataset=ds,
    tokenizer=tokenizer
)


def add_prompt_and_tokenize(row):
    row["input_text"] = f"""# Filename: {row['old_file']}\n# File:\n{row['old_contents']}\n# Instructions:\n{row['inst']}\n# Patch:\n```diff"""
    row["input_ids"] = tokenizer.encode(row["input_text"])
    return row

ds = ds.map(add_prompt_and_tokenize, num_proc=10)

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
