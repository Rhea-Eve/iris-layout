import os
import json
import re
from collections import Counter
from pathlib import Path

# Path to your dataset directory
lib_dir = Path("imaging-extended")

# Match all *_lib.json files
lib_files = list(lib_dir.glob("*_lib.json"))

suffix_counts = Counter()

def extract_suffix(name):
    match = re.search(r"__(.+)", name)
    if not match:
        return None
    suffix = match.group(1).split('_')[0]
    return suffix

for lib_file in lib_files:
    with open(lib_file, "r") as f:
        data = json.load(f)
        for entry in data.values():
            cell_name = entry[2]  # name is the 3rd element
            suffix = extract_suffix(cell_name)
            if suffix:
                suffix_counts[suffix] += 1

# Print results sorted by frequency
for suffix, count in suffix_counts.most_common():
    print(f"{suffix}: {count}")
