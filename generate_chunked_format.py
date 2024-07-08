from generate_ellipsis_format import apply_edits_to_chunks, fix_whitespace_on_chunks
from tree_sitter import Language, Parser # downgrade to version 0.21.3
from tree_sitter_languages import get_language, get_parser
from rapidfuzz.distance.Levenshtein import normalized_similarity
from difflib import unified_diff

parser = get_parser("python")

def find_non_whitespace_indices(s):
    non_whitespace_indices = [i for i, c in enumerate(s) if not c.isspace()]
    if len(non_whitespace_indices) == 0:
        return None, None
    return non_whitespace_indices[0], non_whitespace_indices[-1]

def chunk_text(text: str): # This function should have the same defaults as previous_funct
    tree = parser.parse(bytes(text, 'utf8'))
    lines = text.splitlines()
    split_lines = set([-1, len(lines) - 1])
    
    def get_function_ends(node):
        if node.type in ['class_definition', 'function_definition']:
            # split_lines.add(node.start_point[0]) Adding start point would help separate out imports but would make handling newlines way more difficult
            split_lines.add(node.end_point[0])
        for child in node.children:
            get_function_ends(child)

    get_function_ends(tree.root_node) # We want to automatically split this into chunks.

    chunks = []

    # within each chunk, if the length exceeds, then do the additional split
    split_lines = sorted(list(split_lines))

    for i in range(len(split_lines) - 1):
        print("Processing range: ", split_lines[i] + 1, " to ", split_lines[i + 1], " inclusive.")
        chunk_lines = lines[split_lines[i] + 1:split_lines[i + 1] + 1]
        chunk_text = "\n".join(chunk_lines)

        if len(chunk_lines) > 10:
            print("Trying to split further if possible.")
            chunk_tree = parser.parse(bytes(chunk_text, 'utf8'))
            sub_chunks_lines = set([-1, len(chunk_lines) - 1])
            def get_subchunk_ends(node):
                if node.type in ["for_statement", "while_statement", "if_statement", "block", "elif_clause", "else_clause"]:
                    sub_chunks_lines.add(node.end_point[0])
                for child in node.children:
                    get_subchunk_ends(child)
            get_subchunk_ends(chunk_tree.root_node)
            sub_chunks_lines = sorted(list(sub_chunks_lines))

            valid_subchunks = [[]]
            for i in range(1, len(sub_chunks_lines)):
                new_chunk = chunk_lines[sub_chunks_lines[i - 1] + 1:sub_chunks_lines[i] + 1]
                if len(new_chunk) + len(valid_subchunks) > 10:
                    valid_subchunks.append([])
                valid_subchunks[-1].extend(new_chunk)

            valid_subchunks = ["\n".join(l) for l in valid_subchunks]
            chunks.extend(valid_subchunks)
        else:
            chunks.append(chunk_text)

    new_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) == 0:
            continue
        new_chunks.append(chunk)
    chunks = new_chunks

    # print("======== GENERATED")
    # print("\n------\n".join(chunks))
    # print("======== ORIGINAL")
    # print(text)
    # print("========")

    single_line_chunks = "".join(chunks).replace("\n", "")

    assert normalized_similarity(single_line_chunks, text.replace("\n", "")) > 0.9

    return chunks

def make_changes(original_code, new_code):
    chunks = chunk_text(original_code)
    chunks = fix_whitespace_on_chunks(chunks)
    
    diff = list(unified_diff(("\n".join(chunks)), new_code.splitlines(), n=1000000))
    edited_chunks = apply_edits_to_chunks(chunks, diff)

    if not("\n".join(edited_chunks) == new_code):
        # print("edited code changes")
        print("\n".join(list(unified_diff("\n".join(edited_chunks).splitlines(), new_code.splitlines(), n=5))))

    return chunks, edited_chunks

def generate_chunk_edits_and_input(original_code, new_code):
    original_code = original_code.rstrip()
    new_code = new_code.rstrip()

    original_chunks, edited_chunks = make_changes(original_code, new_code)

    # print(original_chunks)

    chunked_input = "```\n"
    for chunk_index, original_chunk in enumerate(original_chunks):
        chunked_input += f"# Chunk {chunk_index + 1}\n{original_chunk}\n"
    chunked_input += "```"

    # print(chunked_input)

    chunk_edits = ""
    chunk_numbers_edited = []

    for (chunk_index, (original_chunk, edited_chunk)) in enumerate(zip(original_chunks, edited_chunks)):
        if original_chunk != edited_chunk:
            chunk_edits += f"Chunk {chunk_index + 1}\n```\n{edited_chunk}\n```\n"
            chunk_numbers_edited.append(chunk_index + 1)

    return chunked_input, chunk_edits, chunk_numbers_edited

def parse_chunk_edits(original_code, edit_str):
    chunks = [s.strip() for s in edit_str.split("Chunk ")]
    original_code_chunks = chunk_text(original_code)
    for chunk in chunks:
        try:
            chunk_number = int(chunk.split("```python")[0].strip())
            code_chunk = chunk.split("```python")[1].split("```")[0]
            original_code_chunks[chunk_number - 1] = code_chunk
        except:
            print("Error processing chunk :(")
    return "\n".join(original_code_chunks)