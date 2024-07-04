from tree_sitter import Language, Parser # downgrade to version 0.21.3
from tree_sitter_languages import get_language, get_parser
from difflib import unified_diff
import json

parser = get_parser("python")
    
def find_non_whitespace_indices(s):
    non_whitespace_indices = [i for i, c in enumerate(s) if not c.isspace()]
    return non_whitespace_indices[0], non_whitespace_indices[-1]

def chunk_text(text: str, min_chunk_lines=3, max_chunk_lines=5): # CREATE NEW
    unsplit_text = text
    start_index, end_index = find_non_whitespace_indices(text)
    start_chunk = text[:start_index]
    end_chunk = "" if end_index == len(text) - 1 else text[end_index + 1:]

    text = text.strip()

    node = parser.parse(bytes(text, "utf8")).root_node # Tree-sitter doesn't seem to care about indentation :)
    lines = text.splitlines()
    if node.end_point[0] - node.start_point[0] + 1 > max_chunk_lines:
        definition_names = [["class_definition"], ["function_definition"], ["for_statement", "while_statement"], ["if_statement"], ["block", "elif_clause", "else_clause"]]
        
        for split_id in definition_names:
            split_lines = set([])
            for child in node.children:
                # print(child.type)
                if child.type in split_id:
                    split_lines.add(child.start_point[0]) # Starting line of the class definition

            split_lines.add(0)
            split_lines.add(len(lines))
            split_lines = list(split_lines)
            # print(split_lines)

            if len(split_lines) > 2:
                chunks = []
                for i in range(len(split_lines) - 1):
                    chunks += chunk_text("\n".join(lines[split_lines[i]:split_lines[i + 1]]))
                chunks[0] = start_chunk + chunks[0]
                chunks[-1] += end_chunk
                return chunks
            
        # Here, there are no further splits that can be made through any of the items
        # Identify a new-line and split after that because that indicates the end of a code "thought". How closely to max_chunk_lines can it be split?
        chunks = [[]]
        for line in lines:
            if len(line.strip()) == 0 and len(chunks[-1]) >= min_chunk_lines:
                chunks.append([]) # Whitespace after the max_chunk_lines? Then, trigger a split
            chunks[-1].append(line)

        if len(chunks) >= 2:
            if len(chunks[-1]) < min_chunk_lines:
                chunks[-2] += chunks[-1]
                chunks = chunks[:-1]

        joined_chunks = ["\n".join(chunk) for chunk in chunks]
        joined_chunks[0] = start_chunk + joined_chunks[0]
        joined_chunks[-1] = joined_chunks[-1] + end_chunk
        return joined_chunks

    return [unsplit_text]

def fix_whitespace_on_chunks(chunks):
    fixed_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) == 0:
            if len(fixed_chunks) == 0:
                fixed_chunks.append("")
            fixed_chunks[-1] += f"{chunk}"
        else:
            fixed_chunks.append(chunk)
    return fixed_chunks

def make_changes(original_code, new_code):
    original_code = original_code.rstrip()
    new_code = new_code.rstrip()

    original_lines = original_code.split("\n")
    new_lines = new_code.split("\n")
    
    diff = list(unified_diff(original_lines, new_lines, n=1000000))
    first_line_index = 0
    for line_idx, line in enumerate(diff):
        if line.count("@@") == 2:
            first_line_index = line_idx + 1
            break

    diff = diff[first_line_index:]

    chunked_text = fix_whitespace_on_chunks(chunk_text(original_code))

    assert "\n".join(chunked_text) == original_code

    # print(len("\n".join(chunked_text)), len(original_lines))

    # print("\n".join(diff))
    original_line_index = 0
    current_chunk_index = 0
    edited_chunks = []
    edited_chunk_lines = []
    original_chunk_lines = []

    for line in diff:
        if line.startswith("+"):
            edited_chunk_lines.append(line[1:])
        else:
            if "\n".join(original_chunk_lines) == chunked_text[current_chunk_index]: # Check that we have seen the current chunk values before moving on 
                edited_chunks.append("\n".join(edited_chunk_lines))
                edited_chunk_lines = []
                original_chunk_lines = []
                current_chunk_index += 1
            if line.startswith(" "):
                edited_chunk_lines.append(line[1:])
            original_chunk_lines.append(line[1:])

            original_line_index += 1
    edited_chunks.append("\n".join(edited_chunk_lines))

    return chunked_text, edited_chunks
    # for (original_chunk, edited_chunk) in zip(chunked_text, edited_chunks):
    #     if original_chunk != edited_chunk:
    #         print(f"ORIGINAL\n{original_chunk}\nEDITED\n{edited_chunk}")

def generate_compressed_output(original_code, new_code):
    original_chunks, edited_chunks = make_changes(original_code, new_code)
    edited_indices = set([])
    for (chunk_index, (original_chunk, edited_chunk)) in enumerate(zip(original_chunks, edited_chunks)):
        if original_chunk != edited_chunk:
            edited_indices.add(chunk_index)
            edited_indices.add(chunk_index - 1)
            if chunk_index == 0:
                edited_indices.add(chunk_index + 1)
    edited_indices = list(edited_indices)
    previous_index = 0
    
    output = ""
    for edited_index in edited_indices:
        if edited_index != previous_index + 1:
            first_line = edited_chunks[edited_index].splitlines()[0]
            whitespace = first_line[:len(first_line) - len(first_line.lstrip())]
            output += f"{whitespace}@@ ... @@\n"
        output += edited_chunks[edited_index] + "\n"
        previous_index = edited_index
    return output

def apply_ellipsis_code(original_code, ellipsis_code):
    ...

if __name__ == "__main__":
    orig_code = open("test_original.txt", "r").read()
    new_code = open("test_new.txt", "r").read()
    # print("\n-----\n".join(fix_whitespace_on_chunks(chunk_text(orig_code))))
    # make_changes(orig_code, new_code)
    print(generate_compressed_output(orig_code, new_code))