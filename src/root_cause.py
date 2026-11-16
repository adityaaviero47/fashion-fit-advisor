"""Root-cause analysis: where do fit problems come from, and what do they cost?

Writes results/root_cause.json.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data import load

OUT = Path(__file__).resolve().parents[1] / "results"

FIT_WORDS = {
    "small": ["tight", "snug", "small", "size up", "sized up", "too short", "couldn't zip",
              "could not zip", "squeeze"],
    "large": ["loose", "big", "large", "size down", "sized down", "too long", "baggy", "gaping"],
}


def share_of_issues_from_top_items(df, frac):
    per_item = df.groupby("item_id")["issue"].sum().sort_values(ascending=False)
    k = max(1, int(len(per_item) * frac))
    return float(per_item.iloc[:k].sum() / per_item.sum())


def main():
    df = load()
    r = {"reviews": len(df), "customers": int(df.user_id.nunique()), "items": int(df.item_id.nunique()),
         "date_range": [str(df.date.min().date()), str(df.date.max().date())]}
    r["fit_issue_rate"] = float(df.issue.mean())
    r["too_small_rate"] = float(df.small.mean())
    r["too_large_rate"] = float(df.large.mean())

    # 1. cost: ratings
    rt = df.groupby("fit")["rating"].agg(["mean", "count"])
    r["mean_rating"] = {k: round(float(v), 2) for k, v in rt["mean"].items()}
    r["share_rated_10"] = {k: round(float(v), 3) for k, v in
                           df.assign(top=df.rating.eq(10)).groupby("fit")["top"].mean().items()}
    r["share_rated_6_or_less"] = {k: round(float(v), 3) for k, v in
                                  df.assign(low=df.rating.le(6)).groupby("fit")["low"].mean().items()}

    # 2. by category (100+ reviews)
    cat = df.groupby("category").agg(n=("issue", "size"), issue=("issue", "mean"),
                                     small=("small", "mean"), large=("large", "mean"))
    cat = cat[cat.n >= 1000].sort_values("issue", ascending=False)
    r["by_category"] = cat.round(3).reset_index().to_dict("records")

    # 3. by size band
    bands = pd.cut(df["size"], [-1, 4, 8, 12, 16, 20, 60],
                   labels=["0-4", "5-8", "9-12", "13-16", "17-20", "21+"])
    sz = df.groupby(bands, observed=True).agg(n=("issue", "size"), issue=("issue", "mean"),
                                              small=("small", "mean"), large=("large", "mean"))
    r["by_size_band"] = sz.round(3).reset_index().rename(columns={"size": "band"}).to_dict("records")

    # 4. by body type
    bt = df.groupby("body_type").agg(n=("issue", "size"), issue=("issue", "mean"))
    r["by_body_type"] = bt.sort_values("issue", ascending=False).round(3).reset_index().to_dict("records")

    # 5. concentration on items
    r["share_of_issues_from_top_10pct_items"] = share_of_issues_from_top_items(df, 0.10)
    r["share_of_reviews_on_top_10pct_items"] = float(
        df.groupby("item_id").size().sort_values(ascending=False)
        .iloc[:int(df.item_id.nunique() * 0.10)].sum() / len(df))
    items = df.groupby("item_id").agg(n=("issue", "size"), small=("small", "mean"), large=("large", "mean"))
    big = items[items.n >= 30]
    r["items_30plus_reviews"] = int(len(big))
    r["items_running_small_share"] = float(((big.small - big.large) >= 0.15).mean())
    r["items_running_large_share"] = float(((big.large - big.small) >= 0.15).mean())

    # 6. repeat customers: does a past fit issue predict the next one?
    d = df.sort_values(["user_id", "date"]).copy()
    d["prev_issue"] = d.groupby("user_id")["issue"].shift(1)
    rep = d.dropna(subset=["prev_issue"])
    r["repeat_reviews"] = int(len(rep))
    r["issue_rate_after_issue"] = float(rep[rep.prev_issue == 1].issue.mean())
    r["issue_rate_after_good_fit"] = float(rep[rep.prev_issue == 0].issue.mean())

    # 7. review text: what customers say when it does not fit
    words = {}
    for side, terms in FIT_WORDS.items():
        sub = df[df.fit == side]
        words[side] = {t: round(float(sub.text.str.contains(re.escape(t)).mean()), 3) for t in terms}
    r["fit_words_in_reviews"] = words
    r["issue_reviews_mentioning_fit_words"] = {
        side: round(float(df[df.fit == side].text.str.contains(
            "|".join(map(re.escape, terms))).mean()), 3) for side, terms in FIT_WORDS.items()}

    OUT.mkdir(exist_ok=True)
    (OUT / "root_cause.json").write_text(json.dumps(r, indent=1, default=float))
    return r


if __name__ == "__main__":
    res = main()
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_category", "by_body_type")},
                     indent=1, default=float))
    print("top categories:", [(c["category"], c["n"], c["issue"]) for c in res["by_category"][:6]])
    print("lowest:", [(c["category"], c["n"], c["issue"]) for c in res["by_category"][-4:]])
    print("body:", [(c["body_type"], c["issue"]) for c in res["by_body_type"]])
