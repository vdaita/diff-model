import diff_utils
import difflib
from dataclasses import dataclass
from typing import Union, Literal
from datasets import load_dataset

@dataclass
class InsertInstruction:
    after_line: int
    code: str

@dataclass
class DeleteInstruction:
    start_line: int
    end_line: int

@dataclass
class ReplaceInstruction:
    search_code: str
    replacement_code: str

@dataclass
class Instruction:
    type: Literal["insert", "delete", "replace"]
    inst: Union[InsertInstruction, DeleteInstruction, ReplaceInstruction]
    line: int


# original_code = open("test_original.txt", "r").read()
# new_code = open("test_new.txt", "r").read()

def generate_instructions(original_code, new_code):
    original_code = ["<TOP/>"] + original_code.splitlines()
    new_code = ["<TOP/>"] + new_code.splitlines()
    diff_lines = list(difflib.unified_diff(original_code, new_code, n=10000000))
    top_index = 0
    for lidx, line in enumerate(diff_lines):
        if "<TOP/>" in line:
            top_index = lidx
            break

    diff_lines = diff_lines[top_index:]

    original_line_count = 0
    original_last_kept_line = 0

    instructions = []
    dlidx = 0
    while dlidx < len(diff_lines): # Diff line index
        # print("Main loop index: ", dlidx)
        # print(f"{original_line_count}: {diff_lines[dlidx]}")
        if diff_lines[dlidx].startswith("+"):
            insertion_lines = []
            while dlidx < len(diff_lines) and diff_lines[dlidx][0] == "+":
                # print("     Insertion loop index: ", diff_lines[dlidx])
                insertion_lines.append(diff_lines[dlidx][1:])
                dlidx += 1
            
            instructions.append(
                Instruction(
                    type="insert",
                    inst=InsertInstruction(
                        after_line=original_last_kept_line,
                        code="\n".join(insertion_lines)
                    ),
                    line=original_last_kept_line
                )
            )
        else:
            if diff_lines[dlidx].startswith("-"): # count the range
                # print("Starting delete range: ", dlidx)
                start_line = original_line_count
                end_line = original_line_count
                while dlidx < len(diff_lines) and diff_lines[dlidx][0] == "-":
                    end_line = original_line_count
                    dlidx += 1
                    original_line_count += 1

                # print(start_line, end_line)
                
                instructions.append(
                    Instruction(
                        type='delete',
                        inst=DeleteInstruction(
                            start_line=start_line,
                            end_line=end_line# Need to account for the shift made by "TOP"
                        ),
                        line=original_last_kept_line# have the deletions come before the insertions
                    )
                )
            else:
                original_last_kept_line = original_line_count
                original_line_count += 1
                dlidx += 1

    # print("\n".join(diff_lines))

    instructions = sorted(instructions, key=lambda x: x.line)
    new_instructions = []

    for inst in instructions: # This loop just makes sure that the there is more than just whitespace in everything.
        if inst.type == "insert" and len(inst.inst.code.strip()) == 0:
            continue

        
        if inst.type == "delete":
            delete_snippet = "\n".join(original_code[inst.inst.start_line:inst.inst.end_line + 1])
            if len(delete_snippet.strip()) == 0:
                continue
        
        new_instructions.append(inst)

    instructions = new_instructions

    new_instructions = []

    inst_i = 0
    while inst_i < len(instructions):
        if (inst_i < len(instructions) - 1) and (instructions[inst_i].line == instructions[inst_i + 1].line) and not(instructions[inst_i].type == instructions[inst_i + 1].type):
            print("Replacing")
            insert_inst = instructions[inst_i] if instructions[inst_i].type == "insert" else instructions[inst_i + 1]
            delete_inst = instructions[inst_i] if instructions[inst_i].type == "delete" else instructions[inst_i + 1]
            delete_snippet = "\n".join(original_code[delete_inst.inst.start_line:delete_inst.inst.end_line + 1])
            new_instructions.append(
                Instruction(
                    type='replace',
                    inst=ReplaceInstruction(
                        delete_snippet,
                        insert_inst.inst.code
                    ),
                    line=instructions[inst_i].line
                )
            )
            inst_i += 2
        else:
            new_instructions.append(instructions[inst_i])
            inst_i += 1

    return new_instructions

dataset = load_dataset("vdaita/editpackft_inst")

def process_row_code(row):
    instructions = generate_instructions(row['old_contents'], row['new_contents'])
    code_xml = ""
    original_lines = ["<TOP/>"] + row["old_contents"].splitlines()

    shot = """## File:
<TOP/>
def multiply(a, b):
    return a * b

def add(a, b):
    sum = a + b
    return sum

## Changes:
1. Remove the multiply function
2. Make the add function more concise by replacing it with only a return statement
3. Add a subtract function

### Response:
<Delete>
def multiply(a, b):
    return a * b
</Delete>
<Replace>
    sum = a + b
    return sum
<With>
    return a + b
</Replace>
<Insert>
def subtract(a, b):
    return a - b
<After>
    sum = a + b
    return sum

</Insert>
    """

    inst = f"""Generate insert-after, delete, and replace blocks to edit the given file according to the user's instruction. Here's an example:
{shot}

## File:
{"\n".join(original_lines)}

## Changes: 
{row['inst']}

"""

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

    new_row = {
        "INSTRUCTION": inst,
        "RESPONSE": code_xml
    }

    return new_row

# def process_row_line(row):
#     instructions = generate_instructions(row['old_contents'], row['new_contents'])
#     lines = row['old_contents'].splitlines()
#     lined_old_contents = "\n".join(f"{str(i + 1)}|{line}" for i, line in enumerate(lines))

#     line_inst = f"""# File:
# {lined_old_contents}

# # Instruction:
# {row['inst']}

# Generate insert-delete patches to fulfill the instruction."""

#     line_xml = ""

#     for instruction in instructions:
#         if instruction.type == "insert":
#             line_xml += f"<Insert>\n{instruction.inst.code}\n<AfterLine>\n{instruction.inst.after_line}\n</Insert>\n"
#         elif instruction.type == "delete":
#             line_xml += f"<Delete>\n<StartLine>{instruction.inst.start_line}</StartLine>\n<EndLine>{instruction.inst.end_line}</EndLine>\n</Delete>\n"
#             # Should have stuck "to" in the middle

#     new_row = {
#         "INSTRUCTION": line_inst,
#         "RESPONSE": line_xml
#     }

#     return new_row

# line_dataset = dataset.map(process_row_line, num_proc=10)
# line_dataset = line_dataset.remove_columns(dataset["train"].column_names)
# line_dataset.push_to_hub("vdaita/editpackft_inst_line")

code_dataset = dataset.map(process_row_code, num_proc=10)
code_dataset = code_dataset.remove_columns(dataset["train"].column_names)
code_dataset.push_to_hub("vdaita/editpackft_inst_code")

# for instruction in generate_instructions(original_code, new_code):
#     if instruction.type == "insert":
#         print(f"After line {instruction.inst.after_line} insert: ")
#         print(instruction.inst.code)
#     elif instruction.type == "delete":
#         print(f"Remove from line {instruction.inst.start_line} to {instruction.inst.end_line}.")
#         print("\n".join(original_code.splitlines()[instruction.inst.start_line:instruction.inst.end_line + 1]))
