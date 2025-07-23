# Prepare input: see hplt_get_valid_urls.py
# Run:
# $ zstdcat be_1_deduplicated.jsonl.zst | python3 hplt_to_text.py - | uniq | python3 hplt_unicalize.py - > texts_hplt_be

import hashlib
from sys import stdin

hs = set()
for line in stdin:
    line = line.strip()
    if not line:
        print()
        continue
    h = hashlib.sha256(line.encode("utf-8")).hexdigest()
    if h in hs:
        continue
    else:
        print(line)
        hs.add(h)
