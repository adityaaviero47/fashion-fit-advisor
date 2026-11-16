"""Size-advice prototype: predict, before checkout, whether a chosen size will
fit, and test it on later reviews the model never saw.

Training data: reviews before 2017. Test data: reviews from 2017 on (69K).
Item statistics for the test period come only from the training period, and
training rows use out-of-fold item statistics, so no row ever sees its own label.

Writes results/advisor.json and results/widget_examples.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data import load, split

OUT = Path(__file__).resolve().parents[1] / "results"
PRIOR = 20          # smoothing strength: an item needs ~20 reviews before its own rate dominates


def band(size):
    return pd.cut(size, [-1, 4, 8, 12, 16, 20, 60], labels=False)


def item_stats(src, keys, g_small, g_large, prefix):
    s = src.groupby(keys).agg(n=("small", "size"), s=("small", "sum"), l=("large", "sum"))
    s[f"{prefix}_small"] = (s.s + PRIOR * g_small) / (s.n + PRIOR)
    s[f"{prefix}_large"] = (s.l + PRIOR * g_large) / (s.n + PRIOR)
    s[f"{prefix}_n"] = s.n
    return s[[f"{prefix}_small", f"{prefix}_large", f"{prefix}_n"]]


def add_item_features(target, source, g_small, g_large):
    out = target.copy()
    for keys, prefix in ((["item_id"], "item"), (["item_id", "sband"], "item_sz"),
                         (["category", "sband"], "cat_sz")):
        st = item_stats(source, keys, g_small, g_large, prefix)
        out = out.join(st, on=keys)
        out[f"{prefix}_small"] = out[f"{prefix}_small"].fillna(g_small)
        out[f"{prefix}_large"] = out[f"{prefix}_large"].fillna(g_large)
        out[f"{prefix}_n"] = out[f"{prefix}_n"].fillna(0)
    med = source.groupby("item_id")["size"].median().rename("item_median_size")
    out = out.join(med, on="item_id")
    out["size_vs_item_median"] = out["size"] - out["item_median_size"]
    return out


def add_user_history(df):
    """Each review sees only that customer's earlier reviews."""
    d = df.sort_values(["user_id", "date"]).copy()
    g = d.groupby("user_id")
    d["user_prev_n"] = g.cumcount()
    d["user_prev_small"] = g["small"].cumsum() - d["small"]
    d["user_prev_large"] = g["large"].cumsum() - d["large"]
    d["user_prev_size_mean"] = (g["size"].cumsum() - d["size"]) / d["user_prev_n"].replace(0, np.nan)
    d["size_vs_user_usual"] = d["size"] - d["user_prev_size_mean"]
    return d.sort_index()


FEATURES = ["size", "sband", "height_in", "weight_lb", "age", "band", "cup", "body_type_code",
            "rented_for_code", "category_code",
            "item_small", "item_large", "item_n", "item_sz_small", "item_sz_large", "item_sz_n",
            "cat_sz_small", "cat_sz_large", "item_median_size", "size_vs_item_median",
            "user_prev_n", "user_prev_small", "user_prev_large", "size_vs_user_usual"]


def main():
    df = load()
    df["sband"] = band(df["size"])
    for col in ("body_type", "rented_for", "category"):
        df[f"{col}_code"] = df[col].astype("category").cat.codes
    df = add_user_history(df)
    train, test = split(df)
    g_small, g_large = train.small.mean(), train.large.mean()

    # training rows: item statistics from strictly earlier reviews of the same item,
    # so no row ever sees its own label (the test period sees training stats only)
    tr_sorted = train.sort_values("date").copy()
    for keys, prefix in ((["item_id"], "item"), (["item_id", "sband"], "item_sz"),
                         (["category", "sband"], "cat_sz")):
        gg = tr_sorted.groupby(keys)
        n = gg.cumcount()
        s = gg["small"].cumsum() - tr_sorted["small"]
        l = gg["large"].cumsum() - tr_sorted["large"]
        tr_sorted[f"{prefix}_small"] = (s + PRIOR * g_small) / (n + PRIOR)
        tr_sorted[f"{prefix}_large"] = (l + PRIOR * g_large) / (n + PRIOR)
        tr_sorted[f"{prefix}_n"] = n
    gsz = tr_sorted.groupby("item_id")["size"]
    tr_sorted["item_median_size"] = gsz.transform(lambda x: x.expanding().median().shift(1))
    tr_sorted["size_vs_item_median"] = tr_sorted["size"] - tr_sorted["item_median_size"]
    train_f = tr_sorted
    test_f = add_item_features(test, train, g_small, g_large)

    y_tr = train_f["fit"].map({"small": 0, "fit": 1, "large": 2}).values
    y_te = test_f["fit"].map({"small": 0, "fit": 1, "large": 2}).values
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=48,
                                           l2_regularization=1.0, random_state=0)
    model.fit(train_f[FEATURES], y_tr)
    p = model.predict_proba(test_f[FEATURES])
    p_small, p_large = p[:, 0], p[:, 2]
    p_issue = p_small + p_large

    res = {"train_reviews": int(len(train_f)), "test_reviews": int(len(test_f)),
           "test_issue_rate": float(test_f.issue.mean()),
           "test_small_rate": float(test_f.small.mean()), "test_large_rate": float(test_f.large.mean()),
           "test_items_seen_in_train": float(test_f.item_n.gt(0).mean())}

    # discrimination vs simple baselines
    base_item = test_f["item_small"] + test_f["item_large"]
    res["auc_issue"] = {"model": roc_auc_score(test_f.issue, p_issue),
                        "item_rate_only": roc_auc_score(test_f.issue, base_item)}
    res["auc_too_small"] = {"model": roc_auc_score(test_f.small, p_small),
                            "item_rate_only": roc_auc_score(test_f.small, test_f.item_small)}
    res["auc_too_large"] = {"model": roc_auc_score(test_f.large, p_large),
                            "item_rate_only": roc_auc_score(test_f.large, test_f.item_large)}

    # targeting: how many fit problems sit in the riskiest X% of orders
    order = np.argsort(-p_issue)
    issues = test_f.issue.values[order]
    for frac in (0.1, 0.2, 0.3):
        k = int(len(order) * frac)
        res[f"issues_in_riskiest_{int(frac * 100)}pct"] = float(issues[:k].sum() / issues.sum())
        res[f"issue_rate_in_riskiest_{int(frac * 100)}pct"] = float(issues[:k].mean())

    # the advice a product page would show, and how often it is right
    margin = 0.10
    up = (p_small - p_large) >= margin
    down = (p_large - p_small) >= margin
    res["advice"] = {
        "margin": margin,
        "size_up_shown": float(up.mean()), "size_down_shown": float(down.mean()),
        "too_small_rate_when_size_up_shown": float(test_f.small.values[up].mean()),
        "too_large_rate_when_size_down_shown": float(test_f.large.values[down].mean()),
        "share_of_too_small_covered": float(test_f.small.values[up].sum() / test_f.small.sum()),
        "share_of_too_large_covered": float(test_f.large.values[down].sum() / test_f.large.sum()),
        "wrong_direction_rate_size_up": float(test_f.large.values[up].mean()),
        "wrong_direction_rate_size_down": float(test_f.small.values[down].mean()),
    }
    # same advice from a size-blind "runs small / runs large" item label
    blind_up = (test_f.item_small - test_f.item_large).values >= margin
    blind_down = (test_f.item_large - test_f.item_small).values >= margin
    res["size_blind_label"] = {
        "too_small_rate_when_size_up_shown": float(test_f.small.values[blind_up].mean()),
        "too_large_rate_when_size_down_shown": float(test_f.large.values[blind_down].mean()),
        "wrong_direction_rate_size_up": float(test_f.large.values[blind_up].mean()),
        "wrong_direction_rate_size_down": float(test_f.small.values[blind_down].mean()),
        "size_up_shown": float(blind_up.mean()), "size_down_shown": float(blind_down.mean()),
    }

    # fair comparison: model advice shown on exactly as many orders as the size-blind label
    res["size_blind_label"]["share_of_too_small_covered"] = float(
        test_f.small.values[blind_up].sum() / test_f.small.sum())
    res["size_blind_label"]["share_of_too_large_covered"] = float(
        test_f.large.values[blind_down].sum() / test_f.large.sum())
    k_up, k_down = int(blind_up.sum()), int(blind_down.sum())
    top_up = np.argsort(-(p_small - p_large))[:k_up]
    top_down = np.argsort(-(p_large - p_small))[:k_down]
    res["model_at_same_coverage"] = {
        "too_small_rate_when_size_up_shown": float(test_f.small.values[top_up].mean()),
        "too_large_rate_when_size_down_shown": float(test_f.large.values[top_down].mean()),
        "share_of_too_small_covered": float(test_f.small.values[top_up].sum() / test_f.small.sum()),
        "share_of_too_large_covered": float(test_f.large.values[top_down].sum() / test_f.large.sum()),
        "wrong_direction_rate_size_up": float(test_f.large.values[top_up].mean()),
        "wrong_direction_rate_size_down": float(test_f.small.values[top_down].mean()),
        "advice_on_items_with_no_history": float((test_f.item_n.values[np.r_[top_up, top_down]] == 0).mean()),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "advisor.json").write_text(json.dumps(res, indent=1, default=float))

    # examples for the product-page widget prototype
    test_f = test_f.assign(p_small=p_small, p_large=p_large)
    ex = []
    pick = test_f[(test_f.item_n >= 50)].sort_values("item_n", ascending=False)
    for item_id in pick.item_id.unique()[:40]:
        rows = test_f[test_f.item_id == item_id]
        by_band = rows.groupby("sband").agg(p_small=("p_small", "mean"), p_large=("p_large", "mean"),
                                            n=("fit", "size"))
        hist = train[train.item_id == item_id]
        hb = hist.groupby(band(hist["size"])).agg(n=("fit", "size"), small=("small", "mean"),
                                                  large=("large", "mean"))
        ex.append({"item_id": item_id, "category": rows.category.iloc[0],
                   "train_reviews": int(len(hist)),
                   "bands": {int(b): {"p_small": float(v.p_small), "p_large": float(v.p_large),
                                      "hist_n": int(hb.n.get(b, 0)),
                                      "hist_small": float(hb.small.get(b, np.nan)),
                                      "hist_large": float(hb.large.get(b, np.nan))}
                             for b, v in by_band.iterrows()}})
    (OUT / "widget_examples.json").write_text(json.dumps(ex, indent=1, default=float))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=float))
