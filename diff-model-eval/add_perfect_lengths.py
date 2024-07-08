from datasets import load_dataset
import tiktoken
from diff_parser import generate_instructions

ds = load_dataset("vdaita/CanItEditResponses", split="test")
tokenizer = tiktoken.get_encoding("cl100k_base")
optimal_length_whole = [len(tokenizer.encode("```python\n" + row['after'] + "\n```")) for row in ds]

after_xml = []
after_xml_length = []

def process_row_code(row):
    instructions = generate_instructions(row['before'], row['after'])
    code_xml = ""
    original_lines = ["<TOP/>"] + row["before"].splitlines()

    for instruction in instructions:
        if instruction.type == "insert":
            start_index = max(0, instruction.inst.after_line - 3)
            previous_snippet = "\n".join(original_lines[start_index:instruction.inst.after_line + 1])
            if len(instruction.inst.code.strip()) == 0:
                continue
            code_xml += f"<Insert>\n{instruction.inst.code}\n<After>\n{previous_snippet}\n</Insert>\n"
        elif instruction.type == "delete":
            delete_snippet = "\n".join(original_lines[instruction.inst.start_line:instruction.inst.end_line + 1])
            if len(delete_snippet.strip()) == 0:
                continue
            code_xml += f"<Delete>\n{delete_snippet}\n</Delete>\n"
        elif instruction.type == "replace":
            code_xml += f"<Replace>\n{instruction.inst.search_code}\n<With>\n{instruction.inst.replacement_code}</Replace>\n"

    # row["after_xml"] = code_xml
    # row["after_xml_length"] = len(tokenizer.encode(code_xml))

    # return row
    after_xml.append(code_xml)
    after_xml_length.append(len(tokenizer.encode(code_xml)))

for row in ds:
    process_row_code(row)


# ds = ds.map(process_row_code, num_proc=4)
ds = ds.add_column("after_xml", after_xml)
ds = ds.add_column("after_xml_length", after_xml_length)
ds = ds.add_column("after_length", optimal_length_whole)
ds.push_to_hub("vdaita/CanItEditResponses")