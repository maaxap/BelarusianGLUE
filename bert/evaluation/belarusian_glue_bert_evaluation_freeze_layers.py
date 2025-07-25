# Loosely based on https://huggingface.co/docs/transformers/en/tasks/sequence_classification

# Prerequisites:
# - pip install transformers datasets evaluate accelerate scikit-learn

# Common

import typing as t
from sys import argv
import os
import shutil
import csv
import json

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Config, DebertaV2ForSequenceClassification
import evaluate
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay

MODEL_ID_TO_POSTFIX = {
    "google-bert/bert-base-multilingual-cased": "bert",
    "FacebookAI/xlm-roberta-base": "xlmr",
    "microsoft/mdeberta-v3-base": "deberta",
    "HPLT/hplt_bert_base_be": "hpltbert_be",
}
TMP = "./tmp"
HF_DATASET_ID = "maaxap/BelarusianGLUE"

assert len(argv) == 2, "Usage: python3 belarusian_glue_bert_evaluation_freeze_layers.py <model_id>"
assert argv[1] in MODEL_ID_TO_POSTFIX, "Available models: %s" % list(MODEL_ID_TO_POSTFIX.keys())

BASE_MODEL_ID = argv[1]
POSTFIX = MODEL_ID_TO_POSTFIX[argv[1]]

# Configure here
FREEZE_EMBEDDINGS = True
FREEZE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=TMP,
    learning_rate=2e-5, # TODO: Parameterize
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    push_to_hub=False,
    report_to="none",
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class CustomizedDebertaV2Config(DebertaV2Config):
    def __init__(
            self,
            freeze_embeddings: bool = False,
            freeze_layers: t.Optional[t.List[int]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.freeze_embeddings = freeze_embeddings
        self.freeze_layers = freeze_layers

class CustomizedDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.freeze_embeddings:
            for param in self.deberta.embeddings.parameters():
                param.requires_grad_(False)

        if self.config.freeze_layers is not None:
            for i in self.config.freeze_layers:
                for param in self.deberta.encoder.layer[i].parameters():
                    param.requires_grad_(False)

def finetune_bert_for_text_classification(dataset, title, prepare_text_func, n_runs=1,
                                          additional_examples=[], additional_test_set=None,
                                          show_diagrams=False):
    print("Tokenizing the dataset...")
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(prepare_text_func(e), truncation=True),
        batched=False
    )

    additional_kwargs = {}
    config_cls = AutoConfig
    model_cls = AutoModelForSequenceClassification

    if "deberta-v3" in BASE_MODEL_ID:  
        config_cls = CustomizedDebertaV2Config
        model_cls = CustomizedDebertaV2ForSequenceClassification
        additional_kwargs = {
            "freeze_embeddings": FREEZE_EMBEDDINGS,
            "freeze_layers": FREEZE_LAYERS,
        }

    config = config_cls.from_pretrained(
        BASE_MODEL_ID,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        trust_remote_code=("hplt" in BASE_MODEL_ID),
        **additional_kwargs
    )

    print("Initializing...")
    model = model_cls.from_pretrained(
        BASE_MODEL_ID,
        config=config
    )

    FINETUNED_MODEL_ID = "fine_tuning/%s_%s" % (title, POSTFIX)
    best_metric = 0.0
    for _ in range(n_runs):
        print("Fine-tuning, run %s..." % (_ + 1))
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        if trainer.state.best_metric > best_metric:
            best_metric = trainer.state.best_metric
            trainer.save_model(FINETUNED_MODEL_ID)
        assert os.path.exists(TMP)
        shutil.rmtree(TMP)

    print("Loading the best snapshot...")
    classifier = pipeline(
        "text-classification",
        model=FINETUNED_MODEL_ID,
        tokenizer=AutoTokenizer.from_pretrained(FINETUNED_MODEL_ID),
        trust_remote_code=("hplt" in BASE_MODEL_ID),
        device=0
    )

    if additional_examples:
        print("Running additional examples...")
        for e in additional_examples:
            print("Example: %s" % e)
            print("Output:  %s" % classifier(e))

    def evaluate_quality(test_set, handle="test"):
        print("Evaluating quality on the %s set..." % handle)
        test_predictions = classifier([prepare_text_func(x) for x in test_set])
        with open("%s/%s_predictions.json" % (FINETUNED_MODEL_ID, handle.replace(" ", "_")), "w") as f:
            _ = json.dump(test_predictions, f, indent=2)
        y_true = [x["label"] for x in test_set]
        y_pred = [(0 if x["label"] == "NEGATIVE" else 1) for x in test_predictions]
        y_pred_proba = [(x["score"] * ((-1)**(x["label"] == "NEGATIVE"))) / 2 for x in test_predictions]

        print("Accuracy: %.4f" % accuracy.compute(references=y_true, predictions=y_pred)["accuracy"])
        print("MCC:      %.4f" % matthews_corrcoef(y_true, y_pred))

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, pos_label=1)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=1)
        f_scores = [2*p*r/(p+r) for p, r in zip(precision, recall)]
        best_p, best_r, best_f = max(zip(precision, recall, f_scores), key=lambda t: t[-1])
        print("Best F:   %.4f at P=%.4f, R=%.4f" % (best_f, best_p, best_r))

        if show_diagrams:
            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()

    _ = evaluate_quality(dataset["test"])
    if additional_test_set:
        _ = evaluate_quality(additional_test_set, handle="out-of-domain test")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# BeSLS

besls = load_dataset(HF_DATASET_ID, "besls")

_ = finetune_bert_for_text_classification(
    besls, "besls",
    lambda x: x["sentence"],
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# BelaCoLA

belacola = load_dataset(HF_DATASET_ID, "belacola_in_domain")

belacola_ood = load_dataset(HF_DATASET_ID, "belacola_out_of_domain")

_ = finetune_bert_for_text_classification(
    belacola, "belacola",
    lambda x: x["sentence"],
    additional_test_set=belacola_ood["test"]
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# BeWiC

bewic = load_dataset(HF_DATASET_ID, "bewic")

_ = finetune_bert_for_text_classification(
    bewic, "bewic",
    lambda x: "%s | %s | %s" % (x["word"], x["sentence1"], x["sentence2"]),
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# BeWSC
bewsc = load_dataset(HF_DATASET_ID, "bewsc_as_wnli")

_ = finetune_bert_for_text_classification(
    bewsc, "bewsc",
    lambda x: "%s | %s" % (x["sentence1"], x["sentence2"]),
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# BeRTE-WD
bertewd = load_dataset(HF_DATASET_ID, "bertewd")

_ = finetune_bert_for_text_classification(
    bertewd, "bertewd",
    lambda x: "%s | %s" % (x["text"], x["hypothesis"]),
)
