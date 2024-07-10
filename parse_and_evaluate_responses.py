from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from typing import Literal, Optional
import torch
import typer
from tqdm import tqdm
import re
import os
import json
import diff_utils
import subprocess
import tempfile
from enum import Enum
from typing_extensions import Annotated
import sys
from tiktoken import get_encoding
import difflib
from rapidfuzz.fuzz import partial_token_set_ratio

from generate_ellipsis_format import apply_ellipsis_code
from generate_chunked_format import chunk_text

encoding = get_encoding("cl100k_base")

# exec(compile(open("../diff-model/generate_ellipsis_format.py", "rb").read(), "generate_ellipsis_format.py", 'exec'), globals, locals)

class OutputEnum(str, Enum): # TODO: put all of this in one file
    line = "line"
    ir = "ir"
    whole = "whole"
    udiff = "udiff"
    ellipsis = "ellipsis"
    chunked = "chunked"

def add_line_modifications_to_code(original_code, modification_xml):
    new_code = []
    delete_lines = []
    add_lines = {}

    insert_pattern = re.compile(r'<Insert>(.*?)</Insert>')
    insert_matches = insert_pattern.findall(modification_xml)

    for match in insert_matches:
        try:
            code = match.split("<AfterLine/>")[0]
            line_number = int(match.split("<AfterLine/>")[1].strip())
            add_lines[line_number] = code
        except:
            print("Error parsing insert: " , match)


    delete_pattern = re.compile(r'<Delete>(.*?)</Delete>')
    start_line_pattern = re.compile(r'<StartLine>(.*?)</StartLine>')
    end_line_pattern = re.compile(r'<EndLine>(.*?)</EndLine>')

    delete_matches = delete_pattern.findall(modification_xml)
    for match in delete_matches:
        try:
            start_line_matches =  start_line_pattern.findall(match)
            end_line_matches = end_line_pattern.findall(match)
            start_line_match = int(start_line_matches[0].strip())
            end_line_match = int(end_line_matches[0].strip())
            for i in range(start_line_match, end_line_match + 1):
                delete_lines.append(i)
        except:
            print("Error parsing delete:", match)
            continue
    for index, line in enumerate(original_code.splitlines()):
        if index + 1 in delete_lines:
            continue
        new_code.append(line)
        if index + 1 in add_lines:
            new_code.append(add_lines[index])

    return "\n".join(new_code)
   
def add_ir_modifications_to_code(original_code, modification_xml):
    # Do the similar thing as search-replace blocks
    new_code = "<TOP/>\n" + original_code # Since insertions are happening below, use this to make things easier.

    delete_pattern = re.compile(r'<Delete>(.*?)</Delete>')
    insert_pattern = re.compile(r'<Insert>(.*?)</Insert>')
    replace_pattern = re.compile(r'<Replace>(.*?)</Replace>')

    for match in delete_pattern.findall(modification_xml):
        try:
            delete_lines_match = diff_utils.find_best_match(match, original_code)
            new_code = new_code.replace(delete_lines_match.block, "")
        except:
            print("Error parsing code deletion: ", match)

    for match in insert_pattern.findall(modification_xml):
        try: 
            new_lines = match.split("<After>")[0]
            existing_lines = match.split("<After>")[1]

            existing_lines_match = diff_utils.find_best_match(existing_lines, original_code)
            new_code = new_code.replace(
                existing_lines_match.block,
                existing_lines_match.block + "\n" + new_lines
            )
        except:
            print("Error parsing code modification: ", match)

    for match in replace_pattern.findall(modification_xml):
        try:
            orig_lines = match.split("<With>")[0]
            new_lines = match.split("<With>")[1]
            existing_lines_match = diff_utils.find_best_match(existing_lines, original_code)
            new_code = new_code.replace(
                existing_lines_match.block,
                new_lines
            )
        except:
            print("Error parsing code replacement: ", match)

    return new_code

def apply_chunked_modifications_to_code(original_code, modifications_text):
    chunked_code = chunk_text(original_code)
    split_modifications = modifications_text.split("```")
    # print(json.dumps(split_modifications, indent=4))
    print(len(split_modifications))
    for i in range(0, len(split_modifications) - 1, 2):
        numbers = re.findall(r'\d+', split_modifications[i])
        if not(len(numbers)):
            continue
        idx = int(numbers[-1]) - 1

        if idx >= len(chunked_code):
            continue

        new_code = split_modifications[i + 1].strip('\n')

        most_similar_chunks = []
        for j in range(len(chunked_code)): # use the chunk number as a way to discern between similar items, i.e. __str__ or something other than __init__ since it's at the top
            schunk = chunked_code[j].strip('\n')
            if len(schunk) == 0:
                continue
            # first line similarity
            similarity = partial_token_set_ratio(schunk, new_code) + partial_token_set_ratio(schunk.splitlines()[0], new_code.splitlines()[0])
            if similarity > 130:
                most_similar_chunks.append((abs(idx - j), new_code, j))
        
        if len(most_similar_chunks) > 0:
            most_similar_chunks = sorted(most_similar_chunks)
            chunked_code[most_similar_chunks[0][2]] = most_similar_chunks[0][1]
        else:
            # print("Replacing: \n", chunked_code, "\nwith\n", split_modifications[i + 1])
            chunked_code[idx] = new_code
    return "\n".join(chunked_code)

def extract_code_block_for_direct_modifications(response):
    try:
        return response.split("```python")[1].split("```")[0].strip()
    except:
        # print("Failed to parse code block: ", response)
        try: 
            return response.split("with ```).")[1].strip()
        except:
            return response # This just means that the instruction got stripped away.

def run_python_code_with_timeout(contents, timeout):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name    

    try:
        # Run the python file with a timeout
        result = subprocess.run(
            ["python", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True  # To get the output as a string
        )
        # Return the standard output and error
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "The process timed out.", ""
    except Exception as e:
        return "", str(e)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        

def main(model_type: OutputEnum, use_ds: bool, column: Annotated[Optional[str], typer.Argument()] = None, output_folder: Annotated[Optional[str], typer.Argument()] = None):
    dataset = load_dataset("vdaita/CanItEditResponses", split="test")
    # tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    model_type = model_type.value

    output_data_json = {}
    token_count = 0
    accurate_count = 0

    if not(os.path.exists(output_folder)):
        os.makedirs(output_folder)

    token_count_extracted = []
    
    for row in tqdm(dataset):

        output = ""

        if use_ds:
            output = row[column]
        else:
            file_path = os.path.join(output_folder, f"{row['id']}_direct.txt")
            output_file = open(file_path, "r")
            output = output_file.read()
            output_file.close()
        
        # print(output)
        # Take the output, save it, run it, and then check that it worked
        if not(os.path.exists(output_folder)):
            os.makedirs(output_folder)

        new_code = ""
        if model_type == "line":
            old_contents = f"<TOP/>\n{row['before']}"
            new_code = add_line_modifications_to_code(old_contents, output)
        elif model_type == "ir":
            old_contents = f"<TOP/>\n{row['before']}"
            new_code = add_ir_modifications_to_code(old_contents, output)
        elif model_type == "whole":
            old_contents = row['before']
            new_code = extract_code_block_for_direct_modifications(output)
        elif model_type == "ellipsis":
            extracted = extract_code_block_for_direct_modifications(output)
            changed_before = f"print('Program started')\n{row['before']}\nprint('Program ended')"
            token_count_extracted.append(len(encoding.encode(extracted)))
            new_code = apply_ellipsis_code(changed_before, extracted)
        elif model_type == "chunked":
            new_code = apply_chunked_modifications_to_code(row['before'], output)
            token_count_extracted.append(len(encoding.encode(output))) # TODO: standardize token counting?

        new_code = new_code.replace("<TOP/>", "")

        python_code = new_code + f"\n{row['tests']}\n" + "print('SUCCESS')"
        execution_output, error = run_python_code_with_timeout(python_code, 7)

        print(row['id'], ":", execution_output, error)
        if "IndentationError" in error:
            print(new_code)

        out_file = open(os.path.join(output_folder, f"{row['id']}_processed.txt"), "w+")
        out_file.write(new_code)
        out_file.close()

        out_file = open(os.path.join(output_folder, f"{row['id']}_diff.txt"), "w+")
        out_file.write("\n".join(difflib.unified_diff(new_code.splitlines(), row['after'].splitlines())))
        out_file.close()

        # Evaluate the response length in number of tokens
        # print(len(tokens))

        output_data_json[row["id"]] = {}
        output_data_json[row["id"]]["correct"] = ("SUCCESS" in execution_output)
                
        if "SUCCESS" in execution_output:
            accurate_count += 1
    
    out_file = open(os.path.join(output_folder, "data.json"), "w+")
    out_file.write(json.dumps(output_data_json, indent=4))
    out_file.close()

    print(sum(token_count_extracted)/len(token_count_extracted))
    print("Number correct: ", accurate_count, "/", len(output_data_json))

if __name__ == "__main__":
    typer.run(main)
