from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Literal
import torch
import typer
from tqdm import tqdm
import re
import os
import json
import diff_utils
import subprocess
from enum import Enum
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import generate_chunked_format

load_dotenv(".env")

shot_ir_format = """## File:
<TOP/>
def multiply(a, b):
    return a * b

def add(a, b):
    sum = a + b
    return sum

## Instruction:
1. Remove the multiply function
2. Make the add function more concise by replacing it with only a return statement
3. Add a subtract function

### Response:
<Delete>
def multiply(a, b):
    return a * b
</Delete>
<Replac
    sum = a + b
    return sum
<With>
    return a + b
</Replace>
<Insert>
def subtract(a, b):
    return a - b
<After>
    sum = a + b
    return sum

</Insert>
    """

shot_direct_edit = """## File:
```python
def add(a, b):
    return a + b
```
## Changes:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
### Response:
```python
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```"""

# shot_code_editor = """ ## File:
# ```python
# def add(a, b):
#     return a + b
# ```
# ## Changes:
# Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
# ### Response:
# ```python

# """

class OutputEnum(str, Enum):
    line = "line"
    ir = "ir"
    whole = "whole"
    udiff = "udiff"
    ellipsis = "ellipsis"
    chunked = "chunked"

def main(model_id: str, model_type: OutputEnum, output_folder: str, api: str, col_name: str):
    dataset = load_dataset("vdaita/CanItEditResponses", split="test")
    # pipe = pipeline(model=hf_model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    if api == "hf":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    elif api == "openai":
        model = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    model_type = model_type.value

    if not(os.path.exists(output_folder)):
        os.makedirs(output_folder)

    outputs = []
    outputs_token_length = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for row in tqdm(dataset):
        file_path = os.path.join(output_folder, f"{row['id']}_direct.txt")
        if os.path.exists(file_path):
            continue

        formatted_input = ""
        if model_type == "whole":
            formatted_input += f"""Rewrite the file per the user's desired changes. Here's an example: \n{shot_direct_edit}\n## File:
{row['before']}

## Changes:
{row['instruction_descriptive']}

"""
        elif model_type == "ir":
            formatted_input = f"""Generate insert-after, delete, and replace blocks to edit the given file according to the user's instruction. Here's an example:
{shot_ir_format}

## File:
<TOP/>
{row['before']}

## Changes: 
{row['instruction_descriptive']}

"""
        elif model_type == "ellipsis":
            before = "print('Program start')\n" + row['before'] + "\nprint('Program end')"
            formatted_input = f"""Rewrite the file, using ellipsis (@@ ... @@) to skip over chunks of code that should remain unchanged.

## File:
{before}

## Changes: 
{row['instruction_descriptive']}

"""
        elif model_type == "chunked":
            chunks = generate_chunked_format.chunk_text(row['before'])
            chunks = generate_chunked_format.fix_whitespace_on_chunks(chunks)

            chunked_input = "```\n"
            for chunk_index, original_chunk in enumerate(chunks):
                chunked_input += f"# Chunk {chunk_index + 1}\n{original_chunk}\n"
            chunked_input += "```"

            formatted_input = f"""## File:
{chunked_input}

## Changes:
{row['instruction_descriptive']}

First, list which chunks need to be changed in order to implement the requested changes. Then, rewrite only the necessary chunks in the following format:

Chunk x
```
Rewritten chunk x
```

Chunk y
```
Rewritten chunk y
```
"""
            
        output = ""

        if api == "hf":
            formatted_input = tokenizer.apply_chat_template([{
                "role": "user",
                "content": formatted_input
            }], add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
            # output = model(**formatted_input)
            output = model.generate(formatted_input, max_new_tokens=1000, do_sample=True, top_p=0.95)
            output = output[:, formatted_input.shape[1]:][0]
            outputs_token_length.append(output.shape[0])
            output = tokenizer.decode(output)
            
            outputs.append(output)
        elif api == "openai":
            output = model.chat.completions.create(
                messages=[{"role": "user", "content": formatted_input}],
                max_tokens=1000,
                model=model_id
            )
            output = output.choices[0].message.content

            outputs_token_length.append(len(tokenizer.encode(output)))
            outputs.append(output)

        # output = pipe(formatted_input, do_sample=True, max_new_tokens=500, top_p=0.95, **{"use_cache": True})
        # output = row['outputs']
        # output = output.replace(formatted_input, "")
        out_file = open(file_path, "w+")
        out_file.write(output)
        out_file.close()

    dataset = dataset.remove_columns([f"{col_name}_response", f"{col_name}_count"])
    dataset = dataset.add_column(f"{col_name}_response", outputs)
    dataset = dataset.add_column(f"{col_name}_count", outputs_token_length)
    dataset.push_to_hub("vdaita/CanItEditResponses")

if __name__ == "__main__":
    typer.run(main)
