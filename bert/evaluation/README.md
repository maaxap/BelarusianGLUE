# Basic usage

```
python3 belarusian_glue_bert_evaluation.py google-bert/bert-base-multilingual-cased
```

See other available model identifiers in the source code. For pre-trained mDeBERTa-v3, you have to replace the symlink manually in     `.cache/huggingface/hub/models--microsoft--mdeberta-v3-base/snapshots`.

# Freezing embeddings and layers

```
python3 belarusian_glue_bert_evaluation_freeze_layers.py google-bert/bert-base-multilingual-cased  # etc.
```

Specify the freezing configuration in the source code.

# Transfer learning

Complete the prerequisites specified in the source code of `make_train_sets_transfer.py`, then:
```
python3 make_train_sets_transfer.py
python3 belarusian_glue_bert_evaluation_transfer.py google-bert/bert-base-multilingual-cased  # etc.
```

# Transfer learning with freezing embeddings and layers

Similar to the above but run another script:
```
python3 belarusian_glue_bert_evaluation_transfer_freeze_layers.py google-bert/bert-base-multilingual-cased  # etc.
```

Specify the freezing configuration in the source code.
