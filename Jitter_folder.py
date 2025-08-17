#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import truncnorm

INPUT_DIR   = Path("/Users/amiryasin/Desktop/mix_csvs")
OUTPUT_DIR  = Path("/Users/amiryasin/Desktop/synthetic_csvs")
MULTIPLIER  = 10                                 # synthetic rows per original count of each kind of pitch thrown

# jitter caps (distance from average a new synthetic pitch is "allowed to be")
CAP_PCT = {
    "RelSpeed":          0.05,   # ±5 %
    "SpinRate":          0.08,   # ±8 %
    "InducedVertBreak":  0.40,   # ±40 %
    "HorzBreak":         0.35    # ±25 %
}

SIGMA_FRAC = 0.5            # σ = cap * 0.5   → bell that hugs 0–0.6·cap

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def bell_delta(cap):
    """
    Return one random offset drawn from N(0, σ) truncated at ±cap,
    where σ = cap * SIGMA_FRAC.
    """
    sigma = cap * SIGMA_FRAC
    a, b  = -cap / sigma, cap / sigma
    return truncnorm.rvs(a, b, loc=0, scale=sigma)

for csv_file in INPUT_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)
    synth_rows = []

    for _, row in df.iterrows():
        reps = int(row.get("count", 1)) * MULTIPLIER
        for _ in range(reps):
            r = row.copy()
            for col, cap_pct in CAP_PCT.items():
                if col in r and not pd.isna(r[col]):
                    base = float(r[col])
                    cap  = abs(base) * cap_pct
                    r[col] = base + bell_delta(cap)
            r["count"] = 1
            synth_rows.append(r)

    out_path = OUTPUT_DIR / f"{csv_file.stem}_synthetic.csv"
    pd.DataFrame(synth_rows).to_csv(out_path, index=False)
    print(f"✔️ saved → {out_path}")

print("✅ All synthetic files generated.")

