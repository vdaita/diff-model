from datasets import load_dataset
import tiktoken
import difflib
from generate_chunked_format import generate_compressed_output, apply_ellipsis_code
from rapidfuzz import distance

encoding = tiktoken.get_encoding("cl100k_base")
ds = load_dataset("vdaita/CanItEditResponses", split="test")
# ds = ds.select([11])

print(ds.column_names)

after_udiff = []
after_udiff_length = []
after_ellipsis = []
after_ellipsis_length = []

for idx, row in enumerate(ds):
    print("Processing row: ", row['id'])

    row['before'] = f"print('Program start')\n{row['before']}\nprint('Program end')"
    row['after'] = f"print('Program start')\n{row['after']}\nprint('Program end')"

    # print(row['before'])
    diff = "\n".join(difflib.unified_diff(row['before'].splitlines(), row['after'].splitlines(), n=3))
    diff_length = len(encoding.encode(diff))
    comp_output = generate_compressed_output(row['before'], row['after'])

    print("Comp output")
    print(comp_output)

    print("--")

    print(row['after'])
    print("====== applied ")
    # print(apply_ellipsis_code(row['before'], comp_output))

    score = distance.DamerauLevenshtein.normalized_similarity(row['after'], apply_ellipsis_code(row['before'], comp_output))

    print("> REAL AFTER")
    print(row['after'])

    print("Score: ", score, " Row number: ", idx)
    assert score > 0.95

    ellipsis = "```python\n" + comp_output + "\n```"

    ellipsis_length = len(encoding.encode(ellipsis))

    after_udiff.append(diff)
    after_udiff_length.append(diff_length)
    after_ellipsis.append(ellipsis)
    after_ellipsis_length.append(ellipsis_length)

ds = ds.remove_columns(["after_udiff", "after_udiff_length", "after_ellipsis", "after_ellipsis_length"])

ds = ds.add_column("after_udiff", after_udiff)
ds = ds.add_column("after_udiff_length", after_udiff_length)
ds = ds.add_column("after_ellipsis", after_ellipsis)
ds = ds.add_column("after_ellipsis_length", after_ellipsis_length)

ds.push_to_hub("vdaita/CanItEditResponses")