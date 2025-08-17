#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import math
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────
SYN_DIR   = Path("/Users/amiryasin/Desktop/synthetic_csvs")
MODEL_DIR = Path("/Users/amiryasin/Desktop/tagging_models")
FEATURES  = ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak"]
TARGET    = "RetagPitchType"
MAX_ROWS  = 50_000
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def stratified_cap(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    """Downsample to at most max_rows while preserving class proportions."""
    if max_rows is None or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = []
    for cls, g in df.groupby(label_col, sort=False):
        n = max(1, int(round(len(g) * frac)))
        n = min(n, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    capped = pd.concat(parts, ignore_index=True)
    if len(capped) > max_rows:
        capped = capped.sample(n=max_rows, random_state=seed)
    return capped

trained, skipped, failed = 0, 0, 0

for csv_path in sorted(SYN_DIR.glob("*.csv")):
    print(f"⏳ training on {csv_path.name} …")
    try:
        # ── load & clean ──────────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=FEATURES + [TARGET])

        df = stratified_cap(df, TARGET, MAX_ROWS, RANDOM_SEED)

        if len(df) < 5:
            print(f"   🚫 too few rows after cleaning ({len(df)}), skipped")
            skipped += 1
            continue

        X = df[FEATURES].astype(float)
        le = LabelEncoder()
        y = le.fit_transform(df[TARGET].astype(str))
        n_classes = len(le.classes_)

        if n_classes < 2:
            print(f"   🚫 only one class ({le.classes_[0]}) present, skipped")
            skipped += 1
            continue

        if n_classes == 2:
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=120,
                learning_rate=0.15,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=4,
                random_state=RANDOM_SEED,
            )
        else:
            model = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
                n_estimators=120,
                learning_rate=0.15,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=4,
                random_state=RANDOM_SEED,
            )

        # ── train ────────────────────────────────────────────────────────
        model.fit(X, y)

        # ── save model + encoder ────────────────────────────────────────
        safe_stem = csv_path.stem.replace("_synthetic", "").replace(" ", "_")
        out_path = MODEL_DIR / f"{safe_stem}.joblib"
        joblib.dump((model, le, FEATURES), out_path)
        print(f"✅ saved → {out_path}")
        trained += 1

    except Exception as e:
        print(f"   ❌ failed on {csv_path.name}: {e}")
        failed += 1

print(f"\n🎉 Done. Trained: {trained}  |  Skipped: {skipped}  |  Failed: {failed}")

