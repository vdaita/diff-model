from datasets import load_dataset

ds = load_dataset("vdaita/CanItEditResponses", split="test")
count_columns = [column for column in ds.column_names if "count" in column or "length" in column]

ds = ds.with_format("pandas")
for c in count_columns:
    print(c, ds[c].mean())