from datasets import load_dataset
import tiktoken
import difflib
from generate_ellipsis_format import generate_compressed_output, apply_ellipsis_code, enablePrint
from rapidfuzz import distance

encoding = tiktoken.get_encoding("cl100k_base")
ds = load_dataset("vdaita/CanItEditResponses", split="test")
# ds = ds.select([11])

print(ds.column_names)

after_udiff = []
after_udiff_length = []
after_ellipsis = []
after_ellipsis_length = []
after_chunked_format = []

for idx, row in enumerate(ds):
    print("Processing row: ", row['id'])


    # print(row['before'])
    diff = "\n".join(difflib.unified_diff(row['before'].splitlines(), row['after'].splitlines(), n=3))
    diff_length = len(encoding.encode(diff))

    row['before'] = f"print('Program start')\n{row['before']}\nprint('Program end')"
    row['after'] = f"print('Program start')\n{row['after']}\nprint('Program end')"

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

stats = {"whole": {True: [], False: []}, "udiff": {True: [], False: []}, "ellipsis": {True: [], False: []}}

token_thres = 100

for whole_token, udiff_token, ellipsis_token in zip(ds['after_length'], after_udiff_length, after_ellipsis_length):
    g = whole_token > token_thres
    stats["whole"][g].append(whole_token)
    stats["udiff"][g].append(udiff_token)
    stats["ellipsis"][g].append(ellipsis_token)

for g in [True, False]:
    for x in ["whole", "udiff", "ellipsis"]:
        print(f"{x} - {g}: ", sum(stats[x][g])/len(stats[x][g]))

ds = ds.remove_columns(["after_udiff", "after_udiff_length", "after_ellipsis", "after_ellipsis_length"])

ds = ds.add_column("after_udiff", after_udiff)
ds = ds.add_column("after_udiff_length", after_udiff_length)
ds = ds.add_column("after_ellipsis", after_ellipsis)
ds = ds.add_column("after_ellipsis_length", after_ellipsis_length)

ds.push_to_hub("vdaita/CanItEditResponses")