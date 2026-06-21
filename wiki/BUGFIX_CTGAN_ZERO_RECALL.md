# Bug Report: CTGAN Recall Always 0 Despite Increasing Epochs

**File affected:** `src/main.py`
**Symptom:** CTGAN Recall >50K = 0.0 regardless of epoch count (tested at 10, 50, 150).

---

## Root Cause

Two compounding bugs, both caused by feeding one-hot encoded data to SDV instead of raw categorical data.

### Bug 1 — One-hot encoding applied before SDV

`pd.get_dummies(X_raw)` was called early in the pipeline to produce `X_enc`, and `X_train` was derived from that encoded DataFrame. The CTGAN block then built its training table from `X_train`:

```python
# BEFORE (broken)
train_data_sdv = X_train.copy()   # already one-hot encoded — ~100 binary columns
train_data_sdv["income"] = y_train
```

CTGAN uses a conditional generator that is designed to work with raw categorical columns (e.g. `workclass = "Self-emp-not-inc"`). When it receives one-hot dummies instead, it treats each binary column as an independent numeric feature and loses all categorical structure.

### Bug 2 — `income` detected as numerical, sampled as float

Because `y_train` is an integer array (0/1 from `LabelEncoder`), SDV's metadata auto-detection classified the `income` column as **numerical**, not categorical. CTGAN therefore modelled it as a continuous variable and sampled float values like `0.13` or `0.87`.

The downstream RandomForest was then trained with these float labels. When predicting on the test set, `recall_score(..., pos_label=1)` found zero predictions exactly equal to `1` — producing **Recall = 0.0 for every run**, no matter how many epochs were used.

---

## Fix

Feed the original raw DataFrame (`X_raw`) to SDV, with the original string labels (`y_raw`) for `income`. One-hot encode only *after* sampling.

```python
# AFTER (fixed)
train_data_sdv = X_raw.loc[X_train.index].copy()      # raw categoricals
train_data_sdv["income"] = y_raw.loc[X_train.index].values  # string labels e.g. "<=50K"

metadata = Metadata.detect_from_dataframe(data=train_data_sdv)
sdv_model = CTGANSynthesizer(metadata, epochs=args.epochs, verbose=False)
sdv_model.fit(train_data_sdv)

synthetic_sdv = sdv_model.sample(num_rows=len(train_data_sdv))

# Convert string income labels back to 0/1 integers
y_synth_sdv = le.transform(synthetic_sdv["income"].str.strip())

# One-hot encode synthetic features and align columns with the real test set
X_synth_raw = synthetic_sdv.drop("income", axis=1)
X_synth_sdv = pd.get_dummies(X_synth_raw).reindex(columns=X_train.columns, fill_value=0)
```

**What each change does:**

| Change | Why |
|---|---|
| `X_raw.loc[X_train.index]` | Sends raw categorical columns to CTGAN so it can use its conditional generator correctly |
| `y_raw.loc[X_train.index]` | Sends string labels so SDV detects `income` as categorical, not numerical |
| `le.transform(...str.strip())` | Converts sampled string labels back to 0/1 integers for sklearn |
| `.reindex(columns=X_train.columns, fill_value=0)` | Aligns one-hot columns between synthetic and real data (handles unseen categories) |

---

## Lesson

SDV/CTGAN must receive data in its **natural form** — categorical columns as strings, not pre-encoded binary dummies. Pre-encoding before any learned generator (SDV, TVAE, etc.) is a common mistake that silently destroys the generator's ability to model the data correctly. Always encode *after* sampling.
