"""A/B test sizing for the size-advice widget.

Two-proportion test on the fit-issue rate (reviews saying too small or too
large). Writes results/experiment.json.
"""
import json
from pathlib import Path

from scipy.stats import norm

OUT = Path(__file__).resolve().parents[1] / "results"


def n_per_arm(p1, p2, alpha=0.05, power=0.80):
    """Customers per arm to detect p1 -> p2 (two-sided)."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * (2 * pbar * (1 - pbar)) ** 0.5 + zb * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5) ** 2
    return int(num / (p1 - p2) ** 2) + 1


def main():
    adv = json.loads((OUT / "advisor.json").read_text())
    base = adv["test_issue_rate"]
    rows = []
    for drop_pp in (0.5, 1.0, 1.5, 2.0):
        p2 = base - drop_pp / 100
        rows.append({"baseline_issue_rate": round(base, 4), "target_issue_rate": round(p2, 4),
                     "drop_pp": drop_pp, "relative_drop": round(drop_pp / 100 / base, 3),
                     "orders_per_arm": n_per_arm(base, p2)})
    res = {"metric": "fit-issue rate per reviewed order",
           "design": "50/50 split by customer ID, so one customer never sees both versions",
           "alpha": 0.05, "power": 0.80, "sizing": rows}
    (OUT / "experiment.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
