#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python3
"""
Tag pitches in each game CSV using the matching per-pitcher XGBoost model.

Expected layout:
- MODEL_DIR: .joblib files, one per pitcher. Each file is either:
    (model, label_encoder)
    (model, label_encoder, FEATURES)
    {"model":..., "encoder":..., "features":[...]}
- GAME_DIR:  raw game CSVs (one file per game) with a pitcher name column.
- OUT_DIR:   tagged copies written here as <game>_tagged.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
import joblib

# ——— paths ————————————————————————————————————————————————
MODEL_DIR = Path("/Users/amiryasin/desktop/tagging_models")
GAME_DIR  = Path("/Users/amiryasin/desktop/all_games_25")
OUT_DIR   = Path("/Users/amiryasin/desktop/all_games_25_tagged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fallback features if a model bundle didn’t save its own list
DEFAULT_FEATURES = ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak"]

# ——— helpers ————————————————————————————————————————————————
def normalise_pitcher(name: str) -> str:
    """
    'Cook, Cole' or 'Cole Cook' -> 'cole_cook'
    """
    if not isinstance(name, str):
        return ""
    name = name.replace(",", " ").strip()
    parts = [p for p in name.split() if p]
    return "_".join(parts).lower()

def load_model_bundle(path: Path) -> Tuple[object, object, Iterable[str]]:
    """
    Load a .joblib and return (model, encoder, features).

    Accepts tuples/lists of len 2 or 3, or a dict with keys like:
    model / encoder / le / label_encoder / features / FEATURES
    """
    obj = joblib.load(path)

    # dict bundle
    if isinstance(obj, dict):
        model = obj.get("model")
        enc   = obj.get("encoder") or obj.get("le") or obj.get("label_encoder")
        feats = obj.get("features") or obj.get("FEATURES") or DEFAULT_FEATURES
        return model, enc, feats

    # tuple / list bundle
    if isinstance(obj, (tuple, list)):
        if len(obj) == 2:
            model, enc = obj
            feats = DEFAULT_FEATURES
        elif len(obj) == 3:
            model, enc, feats = obj
            feats = feats or DEFAULT_FEATURES
        else:
            raise ValueError(f"{path.name}: unsupported bundle length {len(obj)}")
        return model, enc, feats

    raise ValueError(f"{path.name}: unsupported bundle type {type(obj)}")

def find_pitcher_column(df: pd.DataFrame) -> str | None:
    """
    Return the first column that looks like a pitcher name column, else None.
    """
    for c in df.columns:
        if "pitcher" in str(c).lower():
            return c
    return None

# ——— load models —————————————————————————————————————————————
models: dict[str, dict] = {}
for p in sorted(MODEL_DIR.glob("*.joblib")):
    try:
        model, enc, feats = load_model_bundle(p)
        key = p.stem.replace(" ", "_").lower()
        models[key] = {"model": model, "encoder": enc, "features": list(feats)}
    except Exception as e:
        print(f"✖︎ skipped model {p.name}: {e}")

if not models:
    raise SystemExit("No models found in tagging_models/.")

# ——— tag games ——————————————————————————————————————————————
for game_csv in sorted(GAME_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(game_csv, low_memory=False)
    except Exception as e:
        print(f"✖︎ cannot read {game_csv.name}: {e}")
        continue

    pcol = find_pitcher_column(df)
    if pcol is None:
        print(f"— {game_csv.name}: no pitcher column; skipped")
        continue

    # add output column next to metrics if possible, otherwise to the end
    insert_at = len(df.columns)
    for col in reversed(DEFAULT_FEATURES):
        if col in df.columns:
            insert_at = df.columns.get_loc(col) + 1
            break
    if "final_type" not in df.columns:
        df.insert(insert_at, "final_type", np.nan)

    # tag per pitcher group
    for pitcher in df[pcol].dropna().unique():
        # try "first last" and "last first"
        key = normalise_pitcher(pitcher)
        entry = models.get(key)

        if entry is None and isinstance(pitcher, str) and "," in pitcher:
            last, first = [x.strip() for x in pitcher.split(",", 1)]
            key2 = normalise_pitcher(f"{first} {last}")
            entry = models.get(key2)

        if entry is None:
            # no model for this pitcher
            continue

        feats = entry["features"] or DEFAULT_FEATURES
        if not set(feats).issubset(df.columns):
            missing = [c for c in feats if c not in df.columns]
            print(f"• {game_csv.name}: missing {missing}; {pitcher} skipped")
            continue

        mask = (df[pcol] == pitcher)
        X    = df.loc[mask, feats].astype(float)

        # model outputs integer class ids; map to strings via encoder.classes_
        pred_ids = np.asarray(entry["model"].predict(X)).astype(int).ravel()
        enc = entry["encoder"]
        if hasattr(enc, "classes_") and len(getattr(enc, "classes_", [])) > 0:
            classes = np.asarray(enc.classes_)
            df.loc[mask, "final_type"] = classes[pred_ids]
        else:
            # encoder missing or malformed; write ids as fallback
            df.loc[mask, "final_type"] = pred_ids

    out_file = OUT_DIR / f"{game_csv.stem}_tagged.csv"
    try:
        df.to_csv(out_file, index=False)
        print(f"✓ saved {out_file.name}")
    except Exception as e:
        print(f"✖︎ cannot write {out_file.name}: {e}")

print("Done.")

