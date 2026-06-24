import pandas as pd
import numpy as np

from be_great import GReaT
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate(X_raw, y_raw, train_index, X_train_cols, seed, **kwargs):
    """
    GReaT generator (Transformer-based tabular synthesis)
    """

    le = kwargs["le"]
    epochs = kwargs.get("epochs", 1) #50
    target_col = kwargs.get("target_col", "income")

    # ── 1. Training Data vorbereiten ─────────────────────────────
    train_data = X_raw.loc[train_index].copy()
    train_data[target_col] = y_raw.loc[train_index].values

    # optional: string cleanup (wichtig für Adult dataset)
    train_data[target_col] = train_data[target_col].astype(str).str.strip()

    print(f"   Trainiere GReaT ({epochs} epochs) …")

    # ── 2. Modell ────────────────────────────────────────────────
    model = GReaT(
        llm="distilgpt2",  
        epochs=epochs,
        batch_size=32,
        seed=seed
    )

    model.fit(train_data)

    # ── 3. Sampling ───────────────────────────────────────────────
    synthetic = model.sample(n_samples=len(train_data))

    # ── 4. Target encoding ────────────────────────────────────────
    y_synth = le.transform(synthetic[target_col].astype(str).str.strip())

    # ── 5. Features trennen ───────────────────────────────────────
    X_synth_raw = synthetic.drop(columns=[target_col])

    # ── 6. One-hot Encoding (wie CTGAN / SMOTE Pipeline) ─────────
    X_synth = (
        pd.get_dummies(X_synth_raw)
        .reindex(columns=X_train_cols, fill_value=0)
    )

    return X_synth, y_synth