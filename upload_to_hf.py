from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "bigcode/starcoder2-7b"
peft_path = "./checkpoints/final_checkpoint"

model = AutoModelForCausalLM.from_pretrained(base_model)
model.load_adapter(peft_path)

model.push_to_hub("vdaita/diff-starcoder-7b")
