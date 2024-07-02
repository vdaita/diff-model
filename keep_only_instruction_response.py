from datasets import load_dataset

ds = load_dataset("vdaita/editpackft_inst_code")
for split in ["train", "test"]:
    ds[split] = ds[split].remove_columns([col for col in ds[split].column_names if col not in ["INSTRUCTION", "RESPONSE"]])
ds.push_to_hub("vdaita/editpackft_inst_code")