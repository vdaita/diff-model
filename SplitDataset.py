from datasets import load_dataset

ds = load_dataset("vdaita/editpackftmulti_inst")
ds = ds["train"].train_test_split(test_size=500)
ds.push_to_hub("vdaita/editpackftmulti_inst")