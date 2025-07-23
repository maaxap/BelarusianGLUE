# Prepare input: see hplt_get_valid_urls.py
# Run:
# $ zstdcat be_1_deduplicated.jsonl.zst | python3 hplt_to_text.py - | uniq | python3 hplt_unicalize.py - > texts_hplt_be

from sys import stdin
import json
import re

with open("hplt_be_urls_valid") as f:
    urls_valid = {line.strip() for line in f}

for line in stdin:
    d = json.loads(line)
    assert "url" in d and "text" in d and "langs" in d
    base_url = re.sub(r"^https?://", "", d["url"]).split("/")[0]
    if base_url not in urls_valid:
        continue
    paragraphs = d["text"].split("\n")
    langs = d["langs"]
    if len(paragraphs) != len(langs):
        if len(paragraphs) + 1 == len(langs) and langs[0] == "en":
            langs = langs[1:]
        else:
            continue # ~10 pages with larger discrepancies between 'text' and 'langs'
    for p, lang in zip(paragraphs, langs):
        if lang == "be":
            print(p)
    print()
