from datasets import load_dataset
import tiktoken

ds = load_dataset("vdaita/CanItEditResponses", split="test")
tokenizer = tiktoken.get_encoding("cl100k_base")
optimal_length_whole = [len(tokenizer.encode(row["gpt-4-whole_count"])) for row in ds]
ds = ds.remove_columns(["gpt-4-whole_count"])
ds = ds.add_column("gpt-4-whole_count", optimal_length_whole)

print(ds)

ds.push_to_hub("vdaita/CanItEditResponses")