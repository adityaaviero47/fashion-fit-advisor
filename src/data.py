"""Load and clean the Rent the Runway fit-feedback data (Misra, Wan & McAuley, RecSys 2018)."""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "renttherunway_final_data.json.gz"
SPLIT_DATE = pd.Timestamp("2017-01-01")      # train on the past, test on later reviews


def _height_in(h):
    m = re.match(r"(\d+)' (\d+)\"", str(h))
    return int(m[1]) * 12 + int(m[2]) if m else np.nan


def _bust(b):
    m = re.match(r"(\d+)([a-z]+)", str(b))
    if not m:
        return np.nan, np.nan
    cup = "abcdefgh".find(m[2][0])
    return float(m[1]), (cup if cup >= 0 else np.nan)


def load(path=DATA):
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    df = pd.DataFrame(rows).rename(columns={"bust size": "bust_size", "body type": "body_type",
                                            "rented for": "rented_for"})
    df["date"] = pd.to_datetime(df["review_date"], format="%B %d, %Y")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").where(lambda a: a.between(15, 90))
    df["weight_lb"] = pd.to_numeric(df["weight"].str.replace("lbs", ""), errors="coerce")
    df["height_in"] = df["height"].map(_height_in)
    bust = df["bust_size"].map(_bust)
    df["band"] = [b[0] for b in bust]
    df["cup"] = [b[1] for b in bust]
    df["issue"] = (df["fit"] != "fit").astype(int)
    df["small"] = (df["fit"] == "small").astype(int)
    df["large"] = (df["fit"] == "large").astype(int)
    df["text"] = (df["review_summary"].fillna("") + " " + df["review_text"].fillna("")).str.lower()
    return df.sort_values("date").reset_index(drop=True)


def split(df):
    return df[df["date"] < SPLIT_DATE].copy(), df[df["date"] >= SPLIT_DATE].copy()
