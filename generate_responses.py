from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Literal, List
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
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
from sglang.srt.constrained import build_regex_from_object
from pydantic import BaseModel
import re
import time

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
    parallel_chunked = "parallel_chunked"

class InstructionType(str, Enum):
    insert = "insert"
    replace = "replace"

class Change(BaseModel):
    chunk_index: int
    instruction_type: InstructionType
    description: str

class ChangeList(BaseModel):
    changes: List[Change]

@function
def generate_code(s, code, instruction, parallel=False):
    chunked_code = generate_chunked_format.chunk_text(code)
    code_str = "```\n"
    for chunk_index, chunk in enumerate(chunked_code):
        code_str += f"# Chunk {chunk_index + 1}\n"
        code_str += chunk
        code_str += '\n'
    code_str += "```"
    
    s += user(f"""# Code\n{code_str}\n# Change request\n{instruction}\n
# Plan the changes that need to be made in the JSON format to complete the desired instruction. Change as few chunks as possible. State variable/function names as required, but don't make change descriptions overly specific. At most 2 change descriptions per chunk.""")
    s += gen("plan", regex=build_regex_from_object(ChangeList), max_tokens=250)
    print(instruction, s["plan"])
    steps = json.loads(s["plan"])["changes"]
    chunks_edited = list(set([step['chunk_index'] for step in steps]))
    
    if parallel:
        forks = s.fork(len(chunks_edited))
        for i, f in enumerate(forks):
            f += f"Rewritten chunk {chunks_edited[i]}. Do not truncate chunk code for brevity:\n```\n"
            f += gen(f"chunk", max_tokens=256, stop="```")
        for i in range(len(chunks_edited)):
            new_code = forks[i]["chunk"].replace("```", "")
            chunked_code[chunks_edited[i] - 1] = new_code
        return "\n".join(chunked_code)
    

    for chunk_index in chunks_edited:
        s += user(f"Edited chunk {chunk_index}:\n```\n")
        s += gen(f"chunk_{chunk_index}", stop="```", max_tokens=256)
        new_code = s[f"chunk_{chunk_index}"].replace("```", "")
        print("Generated new code:\n", new_code, "\nfor chunk:\n", chunk_index)
        chunked_code[chunk_index - 1] = new_code    
    return "\n".join(chunked_code)


def main(model_id: str, model_type: OutputEnum, output_folder: str, api: str, col_name: str):
    dataset = load_dataset("vdaita/CanItEditResponses", split="test")
    # pipe = pipeline(model=hf_model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    if api == "hf":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    elif api == "openai":
        model = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        tokenizer = tiktoken.get_encoding("cl100k_base")
    elif api == "sglang":
        set_default_backend(RuntimeEndpoint("http://localhost:30000"))

    
    model_type = model_type.value

    if not(os.path.exists(output_folder)):
        os.makedirs(output_folder)

    outputs = []
    outputs_token_length = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()
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
        elif api == "openai":
            output = model.chat.completions.create(
                messages=[{"role": "user", "content": formatted_input}],
                max_tokens=1000,
                model=model_id
            )
            output = output.choices[0].message.content

            outputs_token_length.append(len(tokenizer.encode(output)))
        elif api == "sglang":
            if model_type == "parallel_chunked":
                new_code = generate_code.run(
                    code=row['before'],
                    instruction=row['instruction_descriptive'],
                    parallel=True
                )
                output = new_code.ret_value
            elif model_type == "chunked":
                new_code = generate_code.run(
                    code=row['before'],
                    instruction=row['instruction_descriptive']
                )
                output = new_code.ret_value
        outputs.append(output)
        # output = pipe(formatted_input, do_sample=True, max_new_tokens=500, top_p=0.95, **{"use_cache": True})
        # output = row['outputs']
        # output = output.replace(formatted_input, "")
        out_file = open(file_path, "w+")
        out_file.write(output)
        out_file.close()
    
    print("TIME TAKEN: ", time.time() - start_time)

    if f"{col_name}_response" in dataset.column_names:
        dataset = dataset.remove_columns([f"{col_name}_response"])
    if f"{col_name}_count" in dataset.column_names:
        dataset = dataset.remove_columns([f"{col_name}_count"])

    dataset = dataset.add_column(f"{col_name}_response", outputs)
    dataset = dataset.add_column(f"{col_name}_count", outputs_token_length)
    dataset.push_to_hub("vdaita/CanItEditResponses")

if __name__ == "__main__":
    typer.run(main)
