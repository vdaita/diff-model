from tree_sitter import Language, Parser # downgrade to version 0.21.3
from tree_sitter_languages import get_language, get_parser
from difflib import unified_diff
from rapidfuzz import distance
from rapidfuzz.distance.Levenshtein import normalized_similarity
from dataclasses import dataclass, field
from heapq import heapify, heappush, heappop, heappushpop
from functools import total_ordering
import json
import sys
import os

parser = get_parser("python")

def blockPrint(): # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def find_non_whitespace_indices(s):
    non_whitespace_indices = [i for i, c in enumerate(s) if not c.isspace()]
    if len(non_whitespace_indices) == 0:
        return None, None
    return non_whitespace_indices[0], non_whitespace_indices[-1]

def chunk_text(text: str, min_chunk_lines=3, max_chunk_lines=3, rec_depth=1): # CREATE NEW
    unsplit_text = text
    start_index, end_index = find_non_whitespace_indices(text)
    # print(start_index, end_index)

    if start_index == None:
        return [unsplit_text] # Will be fixed later

    start_chunk = text[:start_index]
    end_chunk = "" if end_index == len(text) - 1 else text[end_index + 1:]

    assert len(start_chunk.strip()) == 0
    assert len(end_chunk.strip()) == 0

    text = text.strip()

    if rec_depth > 10:
        return [unsplit_text]

    node = parser.parse(bytes(text, "utf8")).root_node # Tree-sitter doesn't seem to care about indentation :)
    lines = text.splitlines()
    if node.end_point[0] - node.start_point[0] + 1 > max_chunk_lines:
        definition_names = [["class_definition"], ["function_definition"], ["for_statement", "while_statement"], ["if_statement"], ["block", "elif_clause", "else_clause"]]
        
        for split_id in definition_names:
            # print(">> Chunking by: ", split_id)
            split_lines = set([])
            for child in node.children:
                # print(child.type)
                if child.type in split_id:
                    split_lines.add(child.start_point[0]) # Starting line of the class definition
            # print(">>       Split lines: ", split_lines)

            split_lines.add(0)
            split_lines.add(len(lines))
            split_lines = sorted(list(split_lines))
            # print(split_lines)

            if len(split_lines) > 2:
                chunks = []
                for i in range(len(split_lines) - 1):
                    chunks += chunk_text("\n".join(lines[split_lines[i]:split_lines[i + 1]]), rec_depth=rec_depth + 1)
                chunks[0] = start_chunk + chunks[0]
                chunks[-1] += end_chunk

                # print(f"         > Splitting at {split_id} level.")
                # print("\n-----\n".join(chunks))
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
    blockPrint()

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

    chunked_text = chunk_text(original_code)
    chunked_text = fix_whitespace_on_chunks(chunked_text)

    # if not("\n".join(chunked_text)) == original_code:
    print("\n------\n".join(chunked_text))
    print("Original: ============")
    print(original_code)    
    print("New: ============")
    print(new_code)
    print("Edited chunks: ============")
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
    
    print("\n------\n".join(edited_chunks))

    enablePrint()
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
            edited_indices.add(chunk_index + 1)
    edited_indices = sorted(list(edited_indices))
    previous_index = -1

    # print("Edited indices: ", edited_indices)
    
    output = ""
    for edited_index in edited_indices:
        if edited_index < 0 or edited_index >= len(edited_chunks):
            # print(edited_index, " out of bounds.")
            continue
        if edited_index != previous_index + 1:
            # print("Adding index with @@: ", edited_index)
            # print("Edited chunk: ")
            # print(bytes(edited_chunks[edited_index], "utf8"))
            
            if len(edited_chunks[edited_index].strip()) == 0:
                whitespace = ""
            else:
                first_line = edited_chunks[edited_index].splitlines()[0]
                whitespace = first_line[:len(first_line) - len(first_line.lstrip())]
            output += f"{whitespace}@@ ... @@\n"
        # print("Adding index content: ", edited_index)
        output += edited_chunks[edited_index] + "\n"
        previous_index = edited_index
    
    return output

@total_ordering
@dataclass
class Match:
    norm_distance: float = field(compare=True)
    start_line: int = field(compare=False)
    end_line: int = field(compare=False)

    def __eq__(self, other):
        return self.norm_distance == other.norm_distance

    def __lt__(self, other):
        return self.norm_distance < other.norm_distance
    
    def __gt__(self, other):
        return self.norm_distance > other.norm_distance

def find_best_matches(original_code, query_code, weight_end: bool = False, line_count_tolerance: int = 3, count: int = 1):
    original_lines = original_code.splitlines()
    query_lines = query_code.strip().splitlines()
    best_match = Match(-1, -1, -1)
    
    ql_len = len(query_lines)
    
    for start_line_index in range(len(original_lines)):
        for end_line_index in range(start_line_index, min(start_line_index + len(query_lines) + line_count_tolerance, len(original_lines))):
            # code_chunk = "\n".join(original_lines[start_line_index:end_line_index + 1])
            code_chunk_lines = original_lines[start_line_index:end_line_index + 1]
            ccl_len = len(code_chunk_lines)

            sim_ratio = normalized_similarity("\n".join(code_chunk_lines), "\n".join(query_lines))
            
            weight_lines = 1 # weighting only the last line by this much might be a little risky?
            if weight_end:
                # print("Weight end: ", "\n".join(code_chunk_lines[max(0, ccl_len - 2):ccl_len]), "\n===\n", "\n".join(query_lines[max(0, ql_len - 2):ql_len]))
                sim_ratio += 1*normalized_similarity("\n".join(code_chunk_lines[max(0, ccl_len - weight_lines):ccl_len]), "\n".join(query_lines[max(0, ql_len - weight_lines):ql_len]))
            else:
                sim_ratio += 1*normalized_similarity("\n".join(code_chunk_lines[0:min(ccl_len, weight_lines)]), "\n".join(query_lines[0:min(ql_len, weight_lines)]))
            best_match = max(best_match, Match(sim_ratio, start_line_index, end_line_index))
    return best_match  

def apply_ellipsis_code(original_code, ellipsis_code): # Consider adding starting and finished lines to the code. 
    blockPrint()

    print("Ellipsis code: ")
    print(ellipsis_code)
    
    chunks = [[]]
    for line in ellipsis_code.splitlines():
        line_wo_whitespace = ''.join([ch for ch in line if not(ch.isspace())])
        if line_wo_whitespace == "@@...@@":
            chunks.append([])
        else:
            chunks[-1].append(line)
    chunks = ["\n".join(lines) for lines in chunks if len(("\n".join(lines)).strip()) > 0]

    print("Code edit chunks: ")
    print("\n----\n".join(chunks))

    remaining_original_code = original_code # TODO: slowly reduce the space being searched as you work your way down the file
    changed_code = original_code

    for chunk in chunks:
        print("Processing chunk: ", chunk)
        chunk_lines = chunk.strip().splitlines() # Make sure only significant lines are being considered.
        best_start_chunk = Match(-1, -1, -1)
        best_end_chunk = Match(-1, -1, -1)
        for start_chunk_length in range(2, 7):
            start_chunk = "\n".join(chunk_lines[:max(start_chunk_length, len(chunk_lines))])
            best_start_chunk = max(best_start_chunk, find_best_matches(original_code, start_chunk))
        for end_chunk_length in range(2, 7):
            end_chunk = "\n".join(chunk_lines[max(0, len(chunk_lines) - end_chunk_length):])
            best_end_chunk = max(best_end_chunk, find_best_matches(original_code, end_chunk, weight_end=True))
        
        print("Best start chunk: \n")
        print("\n".join(original_code.splitlines()[best_start_chunk.start_line:best_start_chunk.end_line + 1]))
        print("Best end chunk:\n")
        print("\n".join(original_code.splitlines()[best_end_chunk.start_line:best_end_chunk.end_line + 1]))
        
        original_chunk = "\n".join(original_code.splitlines()[best_start_chunk.start_line:best_end_chunk.end_line + 1])
        print(">>>  REPLACING")
        print(original_chunk)
        print(">>> WITH")
        print(chunk)
        changed_code = changed_code.replace(original_chunk, chunk)
    
    print("Changed code:")
    print(changed_code)

    enablePrint()
    return changed_code
if __name__ == "__main__":
    orig_code = open("test_original.txt", "r").read()
    new_code = open("test_new.txt", "r").read()
    # print("\n-----\n".join(chunk_text(orig_code)))
    # print("\n-----\n".join(fix_whitespace_on_chunks(chunk_text(orig_code))))
    # make_changes(orig_code, new_code)
    compressed_output = generate_compressed_output(orig_code, new_code)
    print(compressed_output)
    print("-----")
    applied_code = apply_ellipsis_code(orig_code, compressed_output)
    print(applied_code)
