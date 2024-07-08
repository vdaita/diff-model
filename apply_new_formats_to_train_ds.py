from datasets import load_dataset
from generate_ellipsis_format import generate_compressed_output, enablePrint, apply_ellipsis_code
from generate_chunked_format import generate_chunk_edits_and_input, parse_chunk_edits
from rapidfuzz import distance
from tiktoken import get_encoding
import datasets

ds = load_dataset("vdaita/editpackft_inst")
print(len(ds["train"]), print(ds["test"]))

encoding = get_encoding("cl100k_base")

def process_row_ellipsis(row):
    row['old_contents'] = f"print('Program start')\n{row['old_contents']}\nprint('Program end')"
    row['new_contents'] = f"print('Program start')\n{row['new_contents']}\nprint('Program end')"

    inst = f"""Rewrite the file, using ellipsis (@@ ... @@) to skip over chunks of code that should remain unchanged.

## File:
{row['old_contents']}

## Changes: 
{row['inst']}

"""
    
    try:
        compressed_output = generate_compressed_output(row['old_contents'], row['new_contents']) 
        score = distance.Levenshtein.normalized_similarity(row['new_contents'], apply_ellipsis_code(row['old_contents'], compressed_output))

        print(compressed_output.count("@@ ... @@"), len(encoding.encode(compressed_output))/len(encoding.encode(row['new_contents'])))

        assert score > 0.95
        return {
            "INSTRUCTION": inst,
            "RESPONSE": "```python\n" + compressed_output + "\n```"
        }
    except Exception:
        enablePrint()
        print("Error parsing") # Occured 90 times for 5000 rows - <2% error rate
        return {
            "INSTRUCTION": inst,
            "RESPONSE": "```python\n" + row['new_contents'] + "\n```"
        }

def process_row_chunked(row):
    # print("Test")
    chunked_input, chunked_edits, chunked_numbers_edited = generate_chunk_edits_and_input(row['old_contents'], row['new_contents'])
    return {
        "INSTRUCTION": f"""## File:
{chunked_input}

## Changes:
{row['inst']}

First, list which chunks need to be changed in order to implement the requested changes. Then, rewrite only the necessary chunks in the following format:

Chunk x
```
Rewritten chunk x
```

Chunk y
```
Rewritten chunk y
```
""",
    "RESPONSE": f"{'Chunks' if len(chunked_numbers_edited) > 1 else 'Chunk'} {', '.join([str(n) for n in chunked_numbers_edited])} need to be edited.\n{chunked_edits}"
    }

orig_cols = ds["train"].column_names
ds = ds.map(process_row_chunked, num_proc=1)
ds = ds.remove_columns(orig_cols)
ds.push_to_hub("vdaita/editpackft_inst_chunked")