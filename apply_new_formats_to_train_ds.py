from datasets import load_dataset
from generate_ellipsis_format import generate_compressed_output, enablePrint, apply_ellipsis_code
from rapidfuzz import distance

ds = load_dataset("vdaita/editpackft_inst")

def process_row_ellipsis(row):
    row['old_contents'] = f"print('Program start')\n{row['old_contents']}\nprint('Program end')"
    row['new_contents'] = f"print('Program start')\n{row['new_contents']}\nprint('Program end')"

    inst = f"""Rewrite the file, using ellipsis to skip over code that should remain unchanged

## File:
{row['old_contents']}

## Changes: 
{row['new_contents']}

"""
    
    try:
        compressed_output = generate_compressed_output(row['old_contents'], row['new_contents']) 
        score = distance.Levenshtein.normalized_similarity(row['new_contents'], apply_ellipsis_code(row['old_contents'], compressed_output))

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

code_dataset = ds.map(process_row_ellipsis, num_proc=10)
code_dataset = code_dataset.remove_columns(ds["train"].column_names)
code_dataset.push_to_hub("vdaita/editpackft_inst_ellipsis")