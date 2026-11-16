"""Tag fit problems from review TEXT alone, for orders with no fit field.

Most marketplaces collect star ratings and free-text reviews or return
comments, but not a structured "too small / fit / too large" field. This
stage checks how well the text alone recovers the customer's own fit label.

Baseline here: TF-IDF + logistic regression, trained on reviews before 2017,
tested on reviews from 2017 on. The GenAI version is `llm_tagger.py`.
The fit words are left in the text on purpose: they are exactly what a
tagger has to read.

Writes results/text_tagger.json and results/llm_eval_sample.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from data import load, split

OUT = Path(__file__).resolve().parents[1] / "results"
LABELS = ["small", "fit", "large"]
SAMPLE_PER_CLASS = 100          # evaluation sample shared with the LLM tagger


def main():
    df = load()
    train, test = split(df)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(train.text), vec.transform(test.text)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xtr, train.fit)
    pred = clf.predict(Xte)

    rep = classification_report(test.fit, pred, labels=LABELS, output_dict=True, zero_division=0)
    res = {"model": "TF-IDF (1-2 grams) + logistic regression",
           "train_reviews": int(len(train)), "test_reviews": int(len(test)),
           "full_test": {"macro_f1": f1_score(test.fit, pred, labels=LABELS, average="macro"),
                         "per_class": {k: {m: round(rep[k][m], 3) for m in ("precision", "recall", "f1-score")}
                                       for k in LABELS}}}

    # balanced evaluation sample, shared with the LLM tagger so both are scored on the same reviews
    rng = np.random.default_rng(7)
    idx = np.concatenate([rng.choice(np.flatnonzero(test.fit.values == k), SAMPLE_PER_CLASS, replace=False)
                          for k in LABELS])
    sample = test.iloc[idx][["item_id", "category", "size", "fit", "review_summary", "review_text"]].copy()
    sample.insert(0, "sample_id", range(len(sample)))
    sample["baseline_pred"] = pred[idx]
    s_pred, s_true = sample.baseline_pred, sample.fit
    srep = classification_report(s_true, s_pred, labels=LABELS, output_dict=True, zero_division=0)
    res["balanced_sample"] = {"n": int(len(sample)),
                              "accuracy": float((s_pred == s_true).mean()),
                              "macro_f1": f1_score(s_true, s_pred, labels=LABELS, average="macro"),
                              "per_class": {k: {m: round(srep[k][m], 3) for m in ("precision", "recall", "f1-score")}
                                            for k in LABELS}}
    OUT.mkdir(exist_ok=True)
    sample.to_csv(OUT / "llm_eval_sample.csv", index=False)
    (OUT / "text_tagger.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=float))
