"""GenAI fit tagger: an LLM (Gemini) reads review text and returns
(1) too small / fit / too large and (2) which body area caused it.

It is scored on the same balanced 300-review sample as the TF-IDF baseline
(results/llm_eval_sample.csv, from text_tagger.py), against each customer's
own fit label, which the model never sees.

Usage:
    export GEMINI_API_KEY=...          # free key: https://aistudio.google.com/apikey
    python llm_tagger.py               # about 15 API calls (20 reviews per call)
    python llm_tagger.py --dry-run     # print the first prompt, call nothing
Optional: GEMINI_MODEL (default gemini-2.5-flash).

Responses are cached in results/llm_cache.json, so a rerun costs nothing.
Writes results/llm_tagger.json.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

OUT = Path(__file__).resolve().parents[1] / "results"
LABELS = ["small", "fit", "large"]
AREAS = ["bust", "waist", "hips", "length", "shoulders", "arms", "overall", "none"]
BATCH = 20
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT = """You label clothing reviews from an online fashion store.

For each review decide how the garment fit the reviewer in the size they wore:
- "small": too small or too tight (they say it was tight, snug, short, they should size up, could not zip, etc.)
- "large": too large or too loose (they say it was big, loose, long, baggy, gaping, they should size down, etc.)
- "fit": it fit as expected, or the review does not say it was too small or too large.

Also give the body area that caused a fit problem: one of {areas}.
Use "none" when the fit is "fit", and "overall" when no single area is named.

Return ONLY a JSON array, one object per review, in the same order:
[{{"id": <id>, "fit": "small|fit|large", "area": "<area>", "reason": "<at most 8 words>"}}]

Reviews:
{reviews}
"""


def build_prompt(rows):
    reviews = "\n".join(
        f'{{"id": {r.sample_id}, "text": {json.dumps((str(r.review_summary) + ". " + str(r.review_text))[:1200])}}}'
        for r in rows.itertuples())
    return PROMPT.format(areas=", ".join(AREAS), reviews=reviews)


def call_gemini(prompt, key, retries=6):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
           f"?key={key}")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0,
                                            "responseMimeType": "application/json"}}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(min(60, 5 * 2 ** attempt))      # rate limit or busy: back off
                continue
            raise SystemExit(f"Gemini API error {e.code}: {e.read().decode()[:400]}")


def parse(text, ids):
    """Parse the model's JSON array; anything missing or invalid becomes None."""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        items = []
    got = {}
    for it in items if isinstance(items, list) else []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        fit = str(it.get("fit", "")).lower().strip()
        area = str(it.get("area", "")).lower().strip()
        got[i] = {"fit": fit if fit in LABELS else None,
                  "area": area if area in AREAS else "overall",
                  "reason": str(it.get("reason", ""))[:80]}
    return {i: got.get(i, {"fit": None, "area": None, "reason": ""}) for i in ids}


def evaluate(sample, tags):
    s = sample.copy()
    s["llm_pred"] = s.sample_id.map(lambda i: tags[str(i)]["fit"] if str(i) in tags else None)
    s["llm_area"] = s.sample_id.map(lambda i: tags[str(i)]["area"] if str(i) in tags else None)
    valid = s.llm_pred.notna()
    v = s[valid]
    rep = classification_report(v.fit, v.llm_pred, labels=LABELS, output_dict=True, zero_division=0)
    brep = classification_report(v.fit, v.baseline_pred, labels=LABELS, output_dict=True, zero_division=0)
    problems = v[v.llm_pred.isin(["small", "large"]) & (v.llm_pred == v.fit)]
    return {
        "model": MODEL, "n": int(len(s)), "parsed": int(valid.sum()),
        "llm": {"accuracy": float((v.llm_pred == v.fit).mean()),
                "macro_f1": f1_score(v.fit, v.llm_pred, labels=LABELS, average="macro"),
                "per_class": {k: {m: round(rep[k][m], 3) for m in ("precision", "recall", "f1-score")}
                              for k in LABELS},
                "confusion_rows_true_cols_pred": confusion_matrix(v.fit, v.llm_pred, labels=LABELS).tolist()},
        "baseline_same_reviews": {"accuracy": float((v.baseline_pred == v.fit).mean()),
                                  "macro_f1": f1_score(v.fit, v.baseline_pred, labels=LABELS,
                                                       average="macro"),
                                  "per_class": {k: {m: round(brep[k][m], 3)
                                                    for m in ("precision", "recall", "f1-score")}
                                                for k in LABELS}},
        "area_of_correctly_tagged_problems": {
            side: dict(Counter(problems[problems.llm_pred == side].llm_area).most_common())
            for side in ("small", "large")},
    }


def main(dry_run=False):
    sample = pd.read_csv(OUT / "llm_eval_sample.csv")
    batches = [sample.iloc[i:i + BATCH] for i in range(0, len(sample), BATCH)]
    if dry_run:
        print(build_prompt(batches[0]))
        return None
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY first (free key: https://aistudio.google.com/apikey)")
    cache_file = OUT / "llm_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    for n, b in enumerate(batches, 1):
        ids = b.sample_id.tolist()
        if all(str(i) in cache for i in ids):
            continue
        print(f"batch {n}/{len(batches)}", flush=True)
        out = parse(call_gemini(build_prompt(b), key), ids)
        cache.update({str(i): v for i, v in out.items()})
        cache_file.write_text(json.dumps(cache, indent=1))
        time.sleep(4)                                   # stay under free-tier rate limits
    res = evaluate(sample, cache)
    (OUT / "llm_tagger.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    r = main(dry_run="--dry-run" in sys.argv)
    if r:
        print(json.dumps(r, indent=1, default=float))
