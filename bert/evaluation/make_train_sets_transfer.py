# Imports

import os
import csv
from random import shuffle
import json
from datasets import load_dataset

# Constants

HF_DATASET_ID = "maaxap/BelarusianGLUE"

TRAIN, DEV, TEST = "train.csv", "dev.csv", "test.csv"
WORD = "word"
SENTENCE = "sentence"
SENTENCE1 = "sentence1"
SENTENCE2 = "sentence2"
OPTION1 = "option1"
OPTION2 = "option2"
TEXT = "text"
HYPOTHESIS = "hypothesis"
TARGET = "target"
ANSWER = "answer"
LABEL = "label"
GOLD_LABEL = "gold_label"

BELACOLA = "belacola_in_domain"
BELACOLA_OOD = "belacola_out_of_domain"
BEWIC = "bewic"
BEWSC = "bewsc_as_wsc"
BERTEWD = "bertewd"

OUTPUT_TRAIN = {k: {} for k in [BELACOLA, BEWIC, BEWSC, BERTEWD]}
OUTPUT_DEV = {k: {} for k in [BELACOLA, BELACOLA_OOD, BEWIC, BEWSC, BERTEWD]}
OUTPUT_TEST = {k: {} for k in [BELACOLA, BELACOLA_OOD, BEWIC, BEWSC, BERTEWD]}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BelaCoLA

# Preliminaries:
# $ wget https://github.com/sjtu-compling/MELA/raw/main/data.zip
# $ unzip data.zip -P 200240
# $ rm data.zip
# $ mv data mela
# $ mkdir -p dutch-cola
# $ wget https://huggingface.co/datasets/GroNLP/dutch-cola/resolve/main/train.csv -O dutch-cola/train.csv
# $ wget https://huggingface.co/datasets/GroNLP/dutch-cola/resolve/main/val.csv -O dutch-cola/val.csv
# $ mkdir -p HuCoLA
# $ wget https://github.com/nytud/HuCOLA/raw/main/data/cola_train.json -O HuCoLA/train.json
# $ wget https://github.com/nytud/HuCOLA/raw/main/data/cola_dev.json -O HuCoLA/dev.json

# Notes:
# - Use train and dev splits from MELA v1.0, as in MELA v1.1 the total amount of train and dev data is smaller
# - Randomly subsample positive instances to maintain 1 : 1 class ratio

PATH_MELA = "mela/v1.0"

def read_mela(path):
    data = []
    with open(path) as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        assert SENTENCE in header and LABEL in header
        sentence_idx = header.index(SENTENCE)
        label_idx = header.index(LABEL)
        for row in csv_reader:
            assert row[label_idx] in ["0", "1"]
            data.append((row[sentence_idx], int(row[label_idx])))
    return data

def read_dutch_cola(path):
    data = []
    with open(path) as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        assert header[2] == "Acceptability" and header[4] == "Sentence"
        for row in csv_reader:
            assert row[2] in ["0", "1"]
            data.append((row[4], int(row[2])))
    return data

def read_hucola(path):
    with open(path) as f:
        d = json.load(f)
    data = []
    for row in d:
        assert "Sent" in row and LABEL in row
        assert row[LABEL] in ["0", "1"]
        data.append((row["Sent"], int(row[LABEL])))
    return data

def read_belacola(name, split):
    data = []

    dataset = load_dataset(HF_DATASET_ID, name)[split]

    for sentence, label in zip(dataset['sentence'], dataset['label']):
        assert label in [0, 1]
        data.append([sentence, label])
    return data

print("Working on BelaCoLA...")

data_mela = []
for lang in os.listdir(PATH_MELA):
    folds = os.listdir(os.path.join(PATH_MELA, lang))
    assert all(x in [TRAIN, DEV, TEST] for x in folds)
    assert DEV in folds
    data_mela += read_mela(os.path.join(PATH_MELA, lang, DEV))
    if TRAIN in folds:
        data_mela += read_mela(os.path.join(PATH_MELA, lang, TRAIN))

data_mela += read_dutch_cola("dutch-cola/train.csv")
data_mela += read_dutch_cola("dutch-cola/val.csv")

data_mela += read_hucola("HuCoLA/train.json")
data_mela += read_hucola("HuCoLA/dev.json")

# Subsample positive instances
data_mela_pos = [x for x in data_mela if x[1] == 1]
data_mela_neg = [x for x in data_mela if x[1] == 0]
assert len(data_mela_pos) > len(data_mela_neg)
_ = shuffle(data_mela_pos)
data_mela_pos = data_mela_pos[:len(data_mela_neg)]
data_mela = data_mela_pos + data_mela_neg
_ = shuffle(data_mela)

# Add BelaCoLA train set
data_mela += read_belacola(BELACOLA, "train")
_ = shuffle(data_mela)

OUTPUT_TRAIN[BELACOLA][SENTENCE], OUTPUT_TRAIN[BELACOLA][LABEL] = zip(*data_mela)
OUTPUT_DEV[BELACOLA][SENTENCE], OUTPUT_DEV[BELACOLA][LABEL] = zip(*read_belacola(BELACOLA, "validation"))
OUTPUT_TEST[BELACOLA][SENTENCE], OUTPUT_TEST[BELACOLA][LABEL] = zip(*read_belacola(BELACOLA, "test"))
OUTPUT_DEV[BELACOLA_OOD][SENTENCE], OUTPUT_DEV[BELACOLA_OOD][LABEL] = zip(*read_belacola(BELACOLA_OOD, "validation"))
OUTPUT_TEST[BELACOLA_OOD][SENTENCE], OUTPUT_TEST[BELACOLA_OOD][LABEL] = zip(*read_belacola(BELACOLA_OOD, "test"))

print("Train: %s | Dev: %s | Test: %s" % (len(OUTPUT_TRAIN[BELACOLA][LABEL]), len(OUTPUT_DEV[BELACOLA][LABEL]), len(OUTPUT_TEST[BELACOLA][LABEL])))
print("Out-of domain dev: %s | Out-of domain test: %s" % (len(OUTPUT_DEV[BELACOLA_OOD][LABEL]), len(OUTPUT_TEST[BELACOLA_OOD][LABEL])))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BeWiC

# Preliminaries:
# $ wget https://pilehvar.github.io/xlwic/data/xlwic_datasets.zip
# $ wget --no-check-certificate https://russiansuperglue.com/tasks/download/RUSSE
# $ unzip xlwic_datasets.zip
# $ mv RUSSE RUSSE.zip
# $ unzip RUSSE.zip
# $ rm -r __MACOSX
# $ rm xlwic_datasets.zip RUSSE.zip
# $ rm xlwic_datasets/*/.DS_Store

# Notes:
# - Use train and dev splits from XLWiC and RUSSE
# - Randomly subsample negative instances in RUSSE to maintain 1 : 1 class ratio

PATH_XLWIC = "xlwic_datasets"

def read_xlwic(path):
    data = []
    with open(path) as f:
        for line in f:
            fields = line.strip().split("\t")
            word = fields[0]
            sentence1, sentence2, label = fields[-3:]
            assert label in ["0", "1"]
            data.append((word, sentence1, sentence2, int(label)))
    return data

def read_russe(path):
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            assert "word" in d and "sentence1" in d and "sentence2" in d and "label" in d
            assert d["label"] in [True, False]
            data.append((d["word"], d["sentence1"], d["sentence2"], int(d["label"])))
    return data

print("Working on BeWiC...")

data_xlwic = []
for fold in ["train", "valid"]:
    filename = "%s_en.txt" % fold
    assert filename in os.listdir(os.path.join(PATH_XLWIC, "wic_english"))
    data_xlwic += read_xlwic(os.path.join(PATH_XLWIC, "wic_english", filename))

for dirname in ["xlwic_wikt", "xlwic_wn"]:
    for subdirname in os.listdir(os.path.join(PATH_XLWIC, dirname)):
        for filename in os.listdir(os.path.join(PATH_XLWIC, dirname, subdirname)):
            if filename.endswith("_train.txt") or filename.endswith("_valid.txt"):
                data_xlwic += read_xlwic(os.path.join(PATH_XLWIC, dirname, subdirname, filename))

assert sum(1 for x in data_xlwic if x[-1] == 1) == sum(1 for x in data_xlwic if x[-1] == 0)

data_russe = []
for fold in ["train", "val"]:
    data_russe += read_russe("RUSSE/%s.jsonl" % fold)

# Subsample negative instances
data_russe_pos = [x for x in data_russe if x[-1] == 1]
data_russe_neg = [x for x in data_russe if x[-1] == 0]
assert len(data_russe_neg) > len(data_russe_pos)
_ = shuffle(data_russe_neg)
data_russe_neg = data_russe_neg[:len(data_russe_pos)]
data_russe = data_russe_pos + data_russe_neg

data_xlwic_russe = data_xlwic + data_russe
_ = shuffle(data_xlwic_russe)


def read_bewic(split):
    dataset = load_dataset(HF_DATASET_ID, BEWIC)[split]
    data = []
    for d in dataset:
        assert "word" in d and "sentence1" in d and "sentence2" in d and "label" in d
        assert d["label"] in [0, 1]
        data.append((d["word"], d["sentence1"], d["sentence2"], d["label"]))
    return data

# Add BeWiC train set
data_xlwic_russe += read_bewic("train")
_ = shuffle(data_xlwic_russe)

OUTPUT_TRAIN[BEWIC][WORD], OUTPUT_TRAIN[BEWIC][SENTENCE1], OUTPUT_TRAIN[BEWIC][SENTENCE2], OUTPUT_TRAIN[BEWIC][LABEL] = zip(*data_xlwic_russe)
OUTPUT_DEV[BEWIC][WORD], OUTPUT_DEV[BEWIC][SENTENCE1], OUTPUT_DEV[BEWIC][SENTENCE2], OUTPUT_DEV[BEWIC][LABEL] = zip(*read_bewic("validation"))
OUTPUT_TEST[BEWIC][WORD], OUTPUT_TEST[BEWIC][SENTENCE1], OUTPUT_TEST[BEWIC][SENTENCE2], OUTPUT_TEST[BEWIC][LABEL] = zip(*read_bewic("test"))

print("Train: %s | Dev: %s | Test: %s" % (len(OUTPUT_TRAIN[BEWIC][LABEL]), len(OUTPUT_DEV[BEWIC][LABEL]), len(OUTPUT_TEST[BEWIC][LABEL])))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BeWSC

# Preliminaries:
# $ wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
# $ wget 'https://drive.usercontent.google.com/download?id=1GQ0Ya_EAVS-NPd-JwCmLPbEP5fOTQJs-&export=download' -O wino_x_data.zip
# $ mkdir -p XWINO
# $ wget https://github.com/yandex-research/crosslingual_winograd/raw/main/dataset.tsv -O XWINO/dataset.tsv
# $ unzip winogrande_1.1.zip
# $ unzip wino_x_data.zip
# $ rm -r __MACOSX
# $ rm winogrande_1.1.zip wino_x_data.zip

# Notes:
# - All datasets are converted to pairs of sentences: sentence 1 – the original sentence text; sentence 2 – the original sentence text in which the pronoun (if available) or "_" sign is replaced with one of the two spans

def read_bewsc(split):
    dataset = load_dataset(HF_DATASET_ID, BEWSC)[split]
    data = []

    for d in dataset:
        assert TEXT in d and LABEL in d and TARGET in d and "span1_text" in d[TARGET] and "span2_text" in d[TARGET] and "span2_index" in d[TARGET]
        sentence = d[TEXT]
        space_inds = [i for i, c in enumerate(sentence) if c == " "]
        pronoun_idx, pronoun_text = d[TARGET]["span2_index"] - 1, d[TARGET]["span2_text"]
        repl = d[TARGET]["span1_text"]
        assert sentence[space_inds[pronoun_idx]+1:].startswith(pronoun_text)
        prefix, suffix = sentence[:space_inds[pronoun_idx]+1], sentence[space_inds[pronoun_idx]+1+len(pronoun_text):]
        data.append((prefix + "_" + suffix, prefix + repl + suffix, d[LABEL]))
    return data

print("Working on BeWSC...")

data_winogrande_winox_xwino = []

with open("winogrande_1.1/train_xl.jsonl") as f:
    for line in f:
        d = json.loads(line)
        assert SENTENCE in d and OPTION1 in d and OPTION2 in d and ANSWER in d
        assert d[SENTENCE].count("_") == 1
        assert d[ANSWER] in ["1", "2"]
        data_winogrande_winox_xwino.append((d[SENTENCE], d[SENTENCE].replace("_", d[OPTION1]), 1 if d[ANSWER] == "1" else 0))
        data_winogrande_winox_xwino.append((d[SENTENCE], d[SENTENCE].replace("_", d[OPTION2]), 1 if d[ANSWER] == "2" else 0))

for lang in ["de", "fr", "ru"]:
    with open("wino_x_data/lm_wino_x/lm_wino_x.en-%s.jsonl" % lang) as f:
        k_context = "context_%s" % lang
        k_option1 = OPTION1 + "_" + lang
        k_option2 = OPTION2 + "_" + lang
        for line in f:
            d = json.loads(line)
            assert k_context in d and k_option1 in d and k_option2 in d and ANSWER in d
            assert d[k_context].count("_") == 1, d
            assert d[ANSWER] in [1, 2], d
            data_winogrande_winox_xwino.append((d[k_context], d[k_context].replace("_", d[k_option1].strip()), 1 if d[ANSWER] == 1 else 0))
            data_winogrande_winox_xwino.append((d[k_context], d[k_context].replace("_", d[k_option2].strip()), 1 if d[ANSWER] == 2 else 0))

with open("XWINO/dataset.tsv") as f:
    csv_reader = csv.reader(f, delimiter="\t")
    for row in csv_reader:
        sentence = row[3]
        assert "_" not in sentence
        sentence_tok = [('"' if x in ("``", "''") else x) for x in json.loads(row[4])]
        assert type(sentence_tok) is list
        i, j = 0, 0
        token_inds = []
        while i < len(sentence) and j < len(sentence_tok):
            if sentence[i:].startswith(sentence_tok[j]):
                token_inds.append(i)
                i += len(sentence_tok[j])
                j += 1
            else:
                i += 1
        assert len(token_inds) == len(sentence_tok)
        pronoun = json.loads(row[5])
        assert type(pronoun[0]) is str and type(pronoun[1]) is list and len(pronoun[1]) == 2
        i, j = pronoun[1]
        assert 0 <= i < len(token_inds) and 0 <= j <= len(token_inds)
        ti = token_inds[i]
        tj = token_inds[j] if j < len(token_inds) else len(sentence)
        while sentence[tj-1] == " ":
            tj -= 1
        assert sentence[ti:tj] == pronoun[0]
        repl = json.loads(row[6])
        assert len(repl) == 2 and all(type(x[0]) is str and type(x[-1]) is bool for x in repl)
        sentence_no_pronoun = sentence[:ti] + "_" + sentence[tj:]
        data_winogrande_winox_xwino.append((sentence_no_pronoun, sentence[:ti] + repl[0][0] + sentence[tj:], int(repl[0][-1])))
        data_winogrande_winox_xwino.append((sentence_no_pronoun, sentence[:ti] + repl[1][0] + sentence[tj:], int(repl[1][-1])))

assert sum(1 for x in data_winogrande_winox_xwino if x[-1] == 0) == sum(1 for x in data_winogrande_winox_xwino if x[-1] == 1)
_ = shuffle(data_winogrande_winox_xwino)

# Add BeWSC train set
data_winogrande_winox_xwino += read_bewsc("train")
_ = shuffle(data_winogrande_winox_xwino)

OUTPUT_TRAIN[BEWSC][SENTENCE1], OUTPUT_TRAIN[BEWSC][SENTENCE2], OUTPUT_TRAIN[BEWSC][LABEL] = zip(*data_winogrande_winox_xwino)
OUTPUT_DEV[BEWSC][SENTENCE1], OUTPUT_DEV[BEWSC][SENTENCE2], OUTPUT_DEV[BEWSC][LABEL] = zip(*read_bewsc("validation"))
OUTPUT_TEST[BEWSC][SENTENCE1], OUTPUT_TEST[BEWSC][SENTENCE2], OUTPUT_TEST[BEWSC][LABEL] = zip(*read_bewsc("test"))

print("Train: %s | Dev: %s | Test: %s" % (len(OUTPUT_TRAIN[BEWSC][LABEL]), len(OUTPUT_DEV[BEWSC][LABEL]), len(OUTPUT_TEST[BEWSC][LABEL])))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BeRTE-WD

# Preliminaries:
# $ wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
# $ wget https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
# $ unzip multinli_1.0.zip
# $ unzip XNLI-1.0.zip
# $ rm -r __MACOSX
# $ rm multinli_1.0.zip XNLI-1.0.zip
# $ rm multinli_1.0/multinli_1.0_train.*

# Notes:
# - Use dev_matched / dev_mismatched sets from MNLI (due to large size of train) and the dev set of XNLI (due to quality issues in machine-translated train)
# - Subsample non-entailed instances

def read_mnli(path):
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            assert SENTENCE1 in d and SENTENCE2 in d and GOLD_LABEL in d
            assert d[GOLD_LABEL] in ["entailment", "neutral", "contradiction", "-"]
            if d[GOLD_LABEL] == "-": # low inter-annotator agreement
                continue
            label = 1 if d[GOLD_LABEL] == "entailment" else 0
            data.append((d[SENTENCE1], d[SENTENCE2], label))
    return data

def read_bertewd(split):
    dataset = load_dataset(HF_DATASET_ID, BERTEWD)[split]
    data = []
    for d in dataset:
        t, h, label = d['text'], d['hypothesis'], d['label']
        assert label in [1, 0]
        data.append((t, h, label))
    return data

print("Working on BeRTE-WD...")

data_mnli_xnli = []
data_mnli_xnli += read_mnli("multinli_1.0/multinli_1.0_dev_matched.jsonl")
data_mnli_xnli += read_mnli("multinli_1.0/multinli_1.0_dev_mismatched.jsonl")
data_mnli_xnli += read_mnli("XNLI-1.0/xnli.dev.jsonl")

# Subsample negative instances
data_mnli_xnli_pos = [x for x in data_mnli_xnli if x[-1] == 1]
data_mnli_xnli_neg = [x for x in data_mnli_xnli if x[-1] == 0]
assert len(data_mnli_xnli_neg) > len(data_mnli_xnli_pos)
_ = shuffle(data_mnli_xnli_neg)
data_mnli_xnli_neg = data_mnli_xnli_neg[:len(data_mnli_xnli_pos)]
data_mnli_xnli = data_mnli_xnli_pos + data_mnli_xnli_neg
_ = shuffle(data_mnli_xnli)

# Add BeRTE-WD train set
data_mnli_xnli += read_bertewd("train")
_ = shuffle(data_mnli_xnli)

OUTPUT_TRAIN[BERTEWD][TEXT], OUTPUT_TRAIN[BERTEWD][HYPOTHESIS], OUTPUT_TRAIN[BERTEWD][LABEL] = zip(*data_mnli_xnli)
OUTPUT_DEV[BERTEWD][TEXT], OUTPUT_DEV[BERTEWD][HYPOTHESIS], OUTPUT_DEV[BERTEWD][LABEL] = zip(*read_bertewd("validation"))
OUTPUT_TEST[BERTEWD][TEXT], OUTPUT_TEST[BERTEWD][HYPOTHESIS], OUTPUT_TEST[BERTEWD][LABEL] = zip(*read_bertewd("test"))

print("Train: %s | Dev: %s | Test: %s" % (len(OUTPUT_TRAIN[BERTEWD][LABEL]), len(OUTPUT_DEV[BERTEWD][LABEL]), len(OUTPUT_TEST[BERTEWD][LABEL])))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Write output

with open("data_train.json", "w") as f:
    _ = json.dump(OUTPUT_TRAIN, f, ensure_ascii=False, indent=2)
with open("data_validation.json", "w") as f:
    _ = json.dump(OUTPUT_DEV, f, ensure_ascii=False, indent=2)
with open("data_test.json", "w") as f:
    _ = json.dump(OUTPUT_TEST, f, ensure_ascii=False, indent=2)
