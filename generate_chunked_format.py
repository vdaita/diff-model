from tree_sitter import Language, Parser # downgrade to version 0.21.3
from tree_sitter_languages import get_language, get_parser
from difflib import unified_diff
import json

parser = get_parser("python")

def chunk_text(text: str, min_chunk_lines=7, max_chunk_lines=10): 
    node = parser.parse(bytes(text, "utf8")).root_node # Tree-sitter doesn't seem to care about indentation :)
    lines = text.splitlines()
    if node.end_point[0] - node.start_point[0] + 1 > max_chunk_lines:
        definition_names = [["class_definition"], ["function_definition"], ["for_statement", "while_statement"], ["if_statement"], ["block", "elif_clause", "else_clause"]]
        
        for split_id in definition_names:
            split_lines = set([])
            for child in node.children:
                print(child.type)
                if child.type in split_id:
                    split_lines.add(child.start_point[0]) # Starting line of the class definition

            split_lines.add(0)
            split_lines.add(len(lines))
            split_lines = list(split_lines)
            print(split_lines)

            if len(split_lines) > 2:
                chunks = []
                for i in range(len(split_lines) - 1):
                    chunks += chunk_text("\n".join(lines[split_lines[i]:split_lines[i + 1]]))
                return chunks
            
        # Here, there are no further splits that can be made through any of the items
        # Identify a new-line and split after that because that indicates the end of a code "thought". How closely to max_chunk_lines can it be split?
        chunks = [[]]
        for line in lines:
            if len(line.strip()) == 0:
                if len(chunks[-1]) >= min_chunk_lines:
                    chunks.append([]) # Whitespace after the max_chunk_lines? Then, trigger a split
                chunks[-1].append(line)
            else:
                chunks[-1].append(line)

        if len(chunks) >= 2:
            if len(chunks[-1]) < min_chunk_lines:
                chunks[-2] += ["\n"] + chunks[-1]
                chunks = chunks[:-1]

        return ["\n".join(chunk) for chunk in chunks]

    return [text]

def fix_whitespace_on_chunks(chunks):
    fixed_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) == 0:
            if len(fixed_chunks) == 0:
                fixed_chunks.append("")
            fixed_chunks[-1] += f"\n{chunk}"
        else:
            fixed_chunks.append(chunk)
    return fixed_chunks

def make_changes(original_code, new_code):
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
    line_number_to_chunk = {}
    line_number_to_chunk[-1] = -1
    current_line_index = 0
    for chunk_idx, chunk in enumerate(chunked_text):
        chunk_lines = chunk.splitlines()
        for line in chunk_lines:
            line_number_to_chunk[current_line_index] = chunk_idx
            current_line_index += 1

    original_line_index = 0
    current_chunk = -1
    edited_chunks = []
    current_chunk_lines = []
    for line in diff:
        if line.startswith("+"):
            current_chunk_lines.append(line[1:])
        else:
            if current_chunk != line_number_to_chunk[line]:
                edited_chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
            if line.startswith(" "):
                current_chunk_lines.append(line[1:])
            original_line_index += 1
    edited_chunks.append("\n".join(current_chunk_lines))

    for (original_chunk, edited_chunk) in zip(chunked_text, edited_chunks):
        if original_chunk != edited_chunk:
            print(f"ORIGINAL\n{original_chunk}\nEDITED\n{edited_chunk}")

if __name__ == "__main__":
    orig_code = open("test_original.txt", "r").read()
    new_code = open("test_new.txt", "r").read()
    print("\n-----\n".join(fix_whitespace_on_chunks(chunk_text(orig_code))))