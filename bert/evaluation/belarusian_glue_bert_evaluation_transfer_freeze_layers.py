# Loosely based on https://huggingface.co/docs/transformers/en/tasks/sequence_classification

# Prerequisites:
# - pip install transformers datasets evaluate accelerate scikit-learn

# Common

from sys import argv
import json
import typing as t

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Config, DebertaV2ForSequenceClassification
import evaluate
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay

TRAIN, DEV, TEST = "train", "validation", "test"

BELACOLA = "belacola_in_domain"
BELACOLA_OOD = "belacola_out_of_domain"
BEWIC = "bewic"
BEWSC = "bewsc_as_wnli"
BERTEWD = "bertewd"

WORD = "word"
SENTENCE = "sentence"
SENTENCE1 = "sentence1"
SENTENCE2 = "sentence2"
TEXT = "text"
HYPOTHESIS = "hypothesis"
LABEL = "label"

HF_DATASET_ID = "maaxap/BelarusianGLUE"

MODEL_ID_TO_POSTFIX = {
    "google-bert/bert-base-multilingual-cased": "bert",
    "FacebookAI/xlm-roberta-base": "xlmr",
    "microsoft/mdeberta-v3-base": "deberta",
    "HPLT/hplt_bert_base_be": "hpltbert_be",
}

assert len(argv) == 2, "Usage: python3 belarusian_glue_bert_evaluation_transfer_freeze_layers.py <model_id>"
assert argv[1] in MODEL_ID_TO_POSTFIX, "Available models: %s" % list(MODEL_ID_TO_POSTFIX.keys())

BASE_MODEL_ID = argv[1]
POSTFIX = MODEL_ID_TO_POSTFIX[argv[1]]

# Configure here
FREEZE_EMBEDDINGS = True
FREEZE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="tmp",
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
            freeze_embeddings: bool = True,
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def finetune_bert_for_text_classification(dataset, title, prepare_text_func, additional_test_set=None):

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
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        trust_remote_code=("hplt" in BASE_MODEL_ID),
        **additional_kwargs
    )

    print("Initializing...")
    
    model = model_cls.from_pretrained(
        BASE_MODEL_ID,
        config=config
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset[TRAIN],
        eval_dataset=tokenized_dataset[DEV],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning...")
    trainer.train()

    print("Saving the best snapshot...")
    FINETUNED_MODEL_ID = "fine_tuning/%s_%s" % (title, POSTFIX)
    trainer.save_model(FINETUNED_MODEL_ID)
    # TODO: Remove ./tmp

    print("Loading the best snapshot...")
    classifier = pipeline(
        "text-classification",
        model=FINETUNED_MODEL_ID,
        tokenizer=AutoTokenizer.from_pretrained(FINETUNED_MODEL_ID),
        trust_remote_code=("hplt" in BASE_MODEL_ID),
        device=0
    )

    def evaluate_quality(test_set, handle=TEST):
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

    _ = evaluate_quality(dataset[TEST])
    if additional_test_set:
        _ = evaluate_quality(additional_test_set, handle="out-of-domain test")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Construct datasets

def sample_train(dataset, n_train=None):
    if n_train is None:
        return dataset
    else:
        assert len({len(v) for v in dataset.values()}) == 1, {len(v) for v in dataset.values()}
        return {k: v[:n_train] for k, v in dataset.items()}

def make_dataset(name, n_train=None):
    return load_dataset(HF_DATASET_ID, name)

belacola = make_dataset(BELACOLA)
belacola_ood = make_dataset(BELACOLA_OOD)
bewic = make_dataset(BEWIC)
bewsc = make_dataset(BEWSC)
bertewd = make_dataset(BERTEWD)

# Fine-tune the model and report results

# Note there is no BeSLS here. Training data generation and transfer learning for BeSLS
# were handled by separate scripts that currently aren't part of this submission.

_ = finetune_bert_for_text_classification(
    belacola, BELACOLA,
    lambda x: x[SENTENCE],
    additional_test_set=belacola_ood[TEST]
)

_ = finetune_bert_for_text_classification(
    bewic, BEWIC,
    lambda x: "%s[SEP]%s[SEP]%s" % (x[WORD], x[SENTENCE1], x[SENTENCE2]),
)

_ = finetune_bert_for_text_classification(
    bewsc, BEWSC,
    lambda x: "%s[SEP]%s" % (x[SENTENCE1], x[SENTENCE2]),
)

_ = finetune_bert_for_text_classification(
    bertewd, BERTEWD,
    lambda x: "%s[SEP]%s" % (x[TEXT], x[HYPOTHESIS]),
)
