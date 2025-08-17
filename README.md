# Pitch Tagging with XGBoost (Per-Pitcher Workflow)

This project provides a pipeline to **generate synthetic pitch data**, **train pitcher-specific XGBoost models**, and **tag pitch types** in game CSVs.

It’s designed for scenarios with limited labeled data per pitcher: we expand each dataset via jittering (controlled noise), train a model for that pitcher, and then apply the saved model to new games.

---

## Repo Scripts

- **`Jitter_folder.py`**  
  Generates synthetic pitch rows from real data by jittering selected metrics.

- **`model_create.py`**  
  Trains an XGBoost model **per pitcher** from the synthetic CSVs and saves each model (with its label encoder and, if present, feature list).

- **`tagger.01.py`**  
  Walks a folder of game CSVs and applies the **matching pitcher model** to tag every pitch for that pitcher. Outputs `_tagged.csv` files.

> ⚠️ Each script defines/uses its **own columns**. There’s no global hard-coded feature list here; follow what the scripts expect.

---

## Quick Start

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Generate Synthetic Data
Adjust input/output paths inside `Jitter_folder.py` if needed, then:
```bash
python src/Jitter_folder.py
```
- **Input**: your raw per-pitcher CSVs (see example schema below).  
- **Output**: `synthetic_csvs/*.csv` (one per pitcher).

### 3) Train Per-Pitcher Models
Point `model_create.py` to the synthetic folder and a models output folder:
```bash
python src/model_create.py
```
- **Output**: `tagging_models/<pitcher>.joblib` bundles containing `(model, encoder[, features])`.

### 4) Tag Game CSVs
Configure `tagger.01.py` with:
- **models folder** → `tagging_models/`
- **games folder** → raw game CSVs (must include a `Pitcher` column or similar)
- **output folder** → where to write `_tagged.csv`

Run:
```bash
python src/tagger.01.py
```
- **Output**: `all_games_25_tagged/<game>_tagged.csv` with a new `final_type` column.

---

## Example Input CSV (Raw Pitches)

Place a small example in `data/examples/example_raw_pitches.csv` to show users the shape of the data you jitter and later train on. A minimal, typical set of columns looks like:

- **Identifier / context**: `Pitcher`, `GameID`, `PitchNo` (free-form; used for filtering/joining)
- **Metrics used by jitter/training** (these vary by your scripts; common ones include):
  - `RelSpeed` – release velocity (mph)
  - `SpinRate` – RPM
  - `InducedVertBreak` – inches
  - `HorzBreak` – inches
  - (If your pipeline needs them: `RelHeight`, `RelSide`, `Extension`, `SpinAxis`, `Tilt_sin`, `Tilt_cos`)
- **Label (for training)**: `RetagPitchType`

> Only include the columns your scripts expect. The example below is intentionally small and safe to commit.

See `data/examples/example_raw_pitches.csv` included in this repo.

---

## Tips & Gotchas

- **Single-class synthetic sets**: if a pitcher’s synthetic CSV ends up with only one label, training is skipped for that pitcher (you can’t fit a classifier with one class).
- **Pitcher name matching**: `tagger.01.py` normalizes names like `"Cook, Cole"` ⇄ `"Cole Cook"` to find the right model file (e.g., `cole_cook.joblib`). If a model is missing, that pitcher’s rows remain untagged.
- **Feature mismatch**: if a game CSV lacks columns required by a specific model, that pitcher is skipped with a clear log message.
- **Performance**: XGBoost `tree_method="hist"` (where used) accelerates training. You can tweak `n_estimators`, `max_depth`, etc., in your scripts.

---

## Development

- Code lives in `src/`.
- Large artifacts (CSV data and `.joblib` models) are ignored by Git (see `.gitignore`).
- Add/modify paths at the top of each script to match your local folders.

---
## Example
- I've attached an exmaple csv of data the jitter code will be run on
- The csv the jitter code should generate when run
- And an example of a game csv when the saved model for both pitchers is available 

## License

Choose what fits your goals:
- **MIT** – simple & permissive (recommended if you want others to reuse).
- **Apache-2.0** – permissive + explicit patent grant.
- **GPL-3.0** – requires derivative projects to also be open-sourced.

If this repo includes an official `LICENSE` file, that file governs usage.

---
