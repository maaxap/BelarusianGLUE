# Prepare texts

```
wget https://data.hplt-project.org/one/monotext/deduplicated/be/be_1.jsonl.zst
mv be_1.jsonl.zst be_1_deduplicated.jsonl.zst
wget https://data.hplt-project.org/one/monotext/cleaned/be/be_1.jsonl.zst
mv be_1.jsonl.zst be_1_cleaned.jsonl.zst

zstdcat be_1_deduplicated.jsonl.zst | jq -r .url | sed -r 's#^https?://##' | cut -d/ -f1 | sort -S1G --parallel=8 | uniq -c | sort -nr > hplt_be_urls_deduplicated
zstdcat be_1_cleaned.jsonl.zst | jq -r .url | sed -r 's#^https?://##' | cut -d/ -f1 | sort -S1G --parallel=8 | uniq -c | sort -nr > hplt_be_urls_cleaned
python3 hplt_get_valid_urls.py

zstdcat be_1_deduplicated.jsonl.zst | python3 hplt_to_text.py - | uniq | python3 hplt_unicalize.py - > texts_hplt_be_uniq
```

# Run mDeBERTa-v3 pre-training

```
mkdir -p deberta; cd deberta
wget https://github.com/microsoft/DeBERTa/archive/refs/heads/master.zip
unzip master.zip
mv DeBERTa-master/DeBERTa ..
mv DeBERTa-master/requirements.txt ../deberta-requirements.txt
cd ..
rm -r deberta
python3 -m pip install -r deberta-requirements.txt
wget https://github.com/microsoft/DeBERTa/raw/master/experiments/language_model/prepare_data.py
wget https://github.com/microsoft/DeBERTa/raw/master/experiments/language_model/rtd_base.json
```

Then change `vocab_size` to 251000 in `rtd_base.json` and run `rtd_mdeberta.sh`.
