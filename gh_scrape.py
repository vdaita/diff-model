from datasets import load_dataset
import requests

ds = load_dataset("vdaita/gh-commits-2022-meta")

def process_commit(row):
    repo_name = row['repos'].split(",")[0]
    url = f"https://github.com/{repo_name}/commit/{row['commit']}.patch"

# Then, find the diff file

# Then, get the raw file that was change

# Then, apply the patch in reverse

# Save as old file, new file