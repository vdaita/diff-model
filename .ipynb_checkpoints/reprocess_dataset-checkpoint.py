from datasets import load_dataset
import re

def process_diff(diff_content):
    # Regular expression to match everything between @@ and @@
    pattern = r'@@.*?@@'
    # Substitute the matched pattern with @@...@@
    processed_content = re.sub(pattern, '@@...@@', diff_content, flags=re.DOTALL)
    
    # Split the content into lines and filter out lines starting with --- and +++
    lines = processed_content.splitlines()
    filtered_lines = [line for line in lines if not (line.startswith('---') or line.startswith('+++'))]
    
    # Join the filtered lines back into a single string
    result = '\n'.join(filtered_lines)
    
    return result

def process_row(row):
    row["trimmed_patch"] = process_diff(row["patch"])
    row["text"] = f"""# File:\n{row['old_contents']}\n# Instructions:\n{row['inst']}\n# Diff patch:\n```diff\n{row['trimmed_patch']}\n```"""
    return row

dataset = load_dataset("vdaita/editpackft_inst")
dataset = dataset.map(process_row, num_proc=10)
dataset.push_to_hub("vdaita/editpackft_inst")