#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
Tag every pitch in every game CSV using the correct per-pitcher XGBoost model.

Assumptions
-----------
• tagging_models/ : each .joblib is either (model, label_encoder)
                    or (model, label_encoder, FEATURES)
                    or {"model":..., "encoder":..., "features":[...]}
• all_games_25/   : folder of raw game CSVs (one file per game)
• Each file has a text column containing the pitcher name (e.g., 'Pitcher')
• We will look for numeric feature columns per model; if a model ships its own
  feature list, we use that; otherwise we fall back to DEFAULT_FEATURES below.

Outputs
-------
Tagged copies in all_games_25_tagged/, each with a new column 'final_type'
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

MODEL_DIR = Path("/Users/amiryasin/desktop/tagging_models")
GAME_DIR  = Path("/Users/amiryasin/desktop/all_games_25")
OUT_DIR   = Path("/Users/amiryasin/desktop/all_games_25_tagged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fallback feature list incase a model didn't save its own list
DEFAULT_FEATURES = ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak"]

# ── utilities ─────────────────────────────────────────────────────────────
def canon_pitcher(name: str) -> str:
    """'Cook, Cole' or 'Cole Cook' → 'cole_cook' (lowercase, single underscore)."""
    if not isinstance(name, str):
        return ""
    name = name.strip().replace(",", " ")
    parts = [p for p in name.split() if p]
    return "_".join(parts).lower()

def load_model_bundle(path: Path):
    obj = joblib.load(path)
    # dict case
    if isinstance(obj, dict):
        model = obj.get("model")
        enc   = obj.get("encoder") or obj.get("le") or obj.get("label_encoder")
        feats = obj.get("features") or obj.get("FEATURES") or DEFAULT_FEATURES
        return model, enc, feats
    # tuple/list case
    if isinstance(obj, (tuple, list)):
        if len(obj) == 2:
            model, enc = obj
            feats = DEFAULT_FEATURES
        elif len(obj) == 3:
            model, enc, feats = obj
            feats = feats or DEFAULT_FEATURES
        else:
            raise ValueError(f"Unsupported model bundle length {len(obj)} in {path.name}")
        return model, enc, feats
    # anything else
    raise ValueError(f"Unsupported model bundle type in {path.name}: {type(obj)}")

def find_pitcher_column(df: pd.DataFrame):
    """Return the first column that looks like a pitcher name column (case-insensitive)."""
    candidates = [c for c in df.columns if "pitcher" in str(c).lower()]
    return candidates[0] if candidates else None

# ── load all pitcher models once ──────────────────────────────────────────
models = {}
for mdl_path in MODEL_DIR.glob("*.joblib"):
    try:
        model, le, feats = load_model_bundle(mdl_path)
        key = mdl_path.stem.replace(" ", "_").lower()
        models[key] = {"model": model, "encoder": le, "features": feats}
    except Exception as e:
        print("could not load {mdl_path.name}: {e}")

if not models:
    raise SystemExit("No models loaded from tagging_models/")

# ── process each game file ────────────────────────────────────────────────
for game_csv in GAME_DIR.glob("*.csv"):
    df = pd.read_csv(game_csv, low_memory=False)

    # find a pitcher column
    pcol = find_pitcher_column(df)
    if pcol is None:
        print("{game_csv.name}: no pitcher column — skipped.")
        continue

    # insert output column right after the last default feature if present; else at end
    insert_at = len(df.columns)
    for col in reversed(DEFAULT_FEATURES):
        if col in df.columns:
            insert_at = df.columns.get_loc(col) + 1
            break
    if "final_type" not in df.columns:
        df.insert(insert_at, "final_type", np.nan)

    # tag each pitcher segment
    for pitcher in df[pcol].dropna().unique():
        # map pitcher name to key variants
        key1 = canon_pitcher(pitcher)  # e.g. "cole_cook"
        entry = models.get(key1)

        # also try swapping first/last if original had comma
        if entry is None and isinstance(pitcher, str) and "," in pitcher:
            last, first = [x.strip() for x in pitcher.split(",", 1)]
            key2 = canon_pitcher(f"{first} {last}")
            entry = models.get(key2)

        if entry is None:
            # no model for this pitcher
            continue

        model = entry["model"]
        le    = entry["encoder"]
        feats = entry["features"] if entry.get("features") else DEFAULT_FEATURES

        # verify the game file has all features needed by this model
        if not set(feats).issubset(df.columns):
            missing = [c for c in feats if c not in df.columns]
            print(f"{game_csv.name}: missing {missing} — {pitcher} skipped.")
            continue

        mask = (df[pcol] == pitcher)
        X    = df.loc[mask, feats].astype(float)

        # predict integers and map to strings robustly (bypass inverse_transform issues)
        pred_int = np.asarray(model.predict(X)).astype(int).ravel()

        # If encoder has classes_ (LabelEncoder), use it; else, leave ints
        if hasattr(le, "classes_") and len(le.classes_) > 0:
            classes = np.asarray(le.classes_)
            pred_str = classes[pred_int]
        else:
            pred_str = pred_int  # fallback (unlikely)

        df.loc[mask, "final_type"] = pred_str

    # save tagged copy
    out_file = OUT_DIR / f"{game_csv.stem}_tagged.csv"
    df.to_csv(out_file, index=False)
    print(f"saved  {out_file}")

print("All games processed (files without pitcher columns were skipped).")

