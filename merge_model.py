from peft import AutoPeftModelForCausalLM

model_id = "vdaita/diff-starcoder-7b"
peft_model = AutoPeftModelForCausalLM.from_pretrained(model_id)
print(type(peft_model))

merged_model = peft_model.merge_and_unload()
print(type(merged_model))

merged_model.save_pretrained("./merged_model")
