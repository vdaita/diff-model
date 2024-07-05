from generate_ellipsis_format import chunk_text, make_changes

def generate_chunk_edits_and_input(original_code, new_code):
    original_chunks, edited_chunks = make_changes(original_code, new_code)

    chunked_input = ""
    for chunk_index, original_chunk in enumerate(original_chunks):
        chunked_input += f"Chunk {chunk_index}\n```python\n{original_chunk}```\n"

    chunk_edits = ""

    for (chunk_index, (original_chunk, edited_chunk)) in enumerate(zip(original_chunks, edited_chunks)):
        if original_chunk != edited_chunk:
            chunk_edits += f"Chunk {chunk_index}\n```python\n{edited_chunk}\n```\n"

    return chunked_input, chunk_edits


def parse_chunk_edits(original_code, edit_str):
    ...