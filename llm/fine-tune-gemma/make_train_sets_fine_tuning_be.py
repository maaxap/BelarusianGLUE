# Imports

import csv
import json

from datasets import Dataset, DatasetDict, load_dataset

# Constants

TRAIN, DEV, TEST = "train", "validation", "test"

WORD = "word"
SENTENCE = "sentence"
SENTENCE1 = "sentence1"
SENTENCE2 = "sentence2"
TEXT = "text"
HYPOTHESIS = "hypothesis"
LABEL = "label"

BESLS = "besls"
BELACOLA = "belacola_in_domain"
BELACOLA_OOD = "belacola_out_of_domain"
BEWIC = "bewic"
BEWSC = "bewsc_as_wnli"
BERTEWD = "bertewd"

HF_DATASET_ID = "maaxap/BelarusianGLUE"

# Functions

def read_besls(name, split):
    dataset = load_dataset(HF_DATASET_ID, name)[split]

    data = []

    for d in dataset:
        label, sentence = d['label'], d['sentence']
        assert label in [0, 1]
        data.append([sentence, label])
    return data


def read_belacola(name, split):
    data = []

    dataset = load_dataset(HF_DATASET_ID, name)[split]

    for sentence, label in zip(dataset['sentence'], dataset['label']):
        assert label in [0, 1]
        data.append([sentence, label])
    return data


def read_bewic(name, split):
    dataset = load_dataset(HF_DATASET_ID, name)[split]
    data = []
    for d in dataset:
        assert "word" in d and "sentence1" in d and "sentence2" in d and "label" in d
        assert d["label"] in [0, 1]
        data.append((d["sentence1"], d["sentence2"], d["word"], d["label"]))
    return data

def read_bewsc_wnli(name, split):
    dataset = load_dataset(HF_DATASET_ID, name)[split]

    data = []
    for d in dataset:
        sentence1, sentence2, label = d['sentence1'], d['sentence2'], d['label']
        assert label in [0, 1]
        data.append((sentence1, sentence2, label))
    return data


def read_bertewd(name, split):
    dataset = load_dataset(HF_DATASET_ID, name)[split]
    data = []
    for d in dataset:
        t, h, label = d['text'], d['hypothesis'], d['label']
        assert label in [1, 0]
        data.append((t, h, label))
    return data


format_besls = lambda row: "Ваша задача – ацаніць настрой сказа. Выберыце адказ '1', калі лічыце, што сказ выражае станоўчую эмоцыю або меркаванне, інакш адкажыце '0'.\n\nСказ: %s\n\nАдказ: %s" % tuple(row)

format_belacola = lambda row: "Ваша задача – вызначыць, ці з’яўляецца сказ прымальным. Выберыце адказ '1', калі вы лічыце, што ў гэтым сказе няма памылак і ён гучыць натуральна (так мог бы сказаць носьбіт беларускай мовы), інакш адкажыце '0'.\n\nСказ: %s\n\nАдказ: %s" % tuple(row)

format_bewic = lambda row: "Перад вамі – пары сказаў, у якіх сустракаецца пэўнае слова. Ваша задача – вызначыць, ці ўжываецца гэтае слова ў пары ў аднолькавых ці ў розных значэннях. Выберыце адказ '1', калі ў сказах прадстаўлена адно ці вельмі блізкія значэнні слова, інакш адкажыце '0'.\n\nСказ 1: %s\n\nСказ 2: %s\n\nСлова: \"%s\"\n\nАдказ: %s" % tuple(row)

format_entailment = lambda row: "Перад вамі пары сказаў. Ваша задача – вызначыць, ці выводзіцца другі сказ з першага. Выберыце адказ '1', калі другі сказ – гэта вывад з першага, інакш адкажыце '0'.\n\nСказ 1: %s\n\nСказ 2: %s\n\nАдказ: %s" % tuple(row)

def make_dataset(name, read_func, format_func):
    to_dataset = lambda split: Dataset.from_dict({
        "text": [
            format_func(x)
            for x in read_func(name, split)
        ]
    })
    return DatasetDict({
        TRAIN: to_dataset("train"),
        DEV: to_dataset("validation"),
        TEST: to_dataset("test")
    })

# Main loop

# Before running, make sure you have the data folder at the required level

besls = make_dataset(
    BESLS,
    read_besls, format_besls
)

belacola_in_domain = make_dataset(
    BELACOLA, read_belacola, format_belacola
)

bewic = make_dataset(
    BEWIC, read_bewic, format_bewic
)

bewsc = make_dataset(
    BEWSC, read_bewsc_wnli, format_entailment
)

bertewd = make_dataset(
    BERTEWD, read_bertewd, format_entailment
)

besls.save_to_disk("./llm_fine_tuning/prompts_be/besls")
belacola_in_domain.save_to_disk("./llm_fine_tuning/prompts_be/belacola_in_domain")
bewic.save_to_disk("./llm_fine_tuning/prompts_be/bewic")
bewsc.save_to_disk("./llm_fine_tuning/prompts_be/bewsc")
bertewd.save_to_disk("./llm_fine_tuning/prompts_be/bertewd")

# Then manually move the .arrow files one level up and delete the folders that are no longer required
