import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import NearestNeighbors


def utility_metrics(y_true, y_pred, label=""):
    """F1 (macro) und Recall der Minderheitsklasse (class=1, d.h. >50K)."""
    f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1,  zero_division=0)
    return {"F1 (macro)": round(f1, 4), "Recall >50K": round(rec, 4)}


def fidelity_metrics(real: pd.DataFrame, synth: pd.DataFrame, cols: list):
    """
    Fidelity: mittlere KS-Statistik über numerische Spalten.
    KS in [0, 1] — niedriger ist besser (synthetische ≈ echte Verteilung).
    """
    ks_scores = []
    for col in cols:
        if col in real.columns and col in synth.columns:
            stat, _ = ks_2samp(real[col].values, synth[col].values)
            ks_scores.append(stat)
    mean_ks = round(float(np.mean(ks_scores)), 4) if ks_scores else float("nan")
    return {"Ø KS-Statistik": mean_ks}


def privacy_nndr(real_train: np.ndarray, synth: np.ndarray, sample_n=2000, seed=42):
    """
    Nearest-Neighbor Distance Ratio (NNDR).
    Für jeden synthetischen Punkt: dist(1-NN_real) / dist(2-NN_real).

    Interpretation
    ──────────────
    NNDR ≈ 1  → 1st und 2nd Neighbor gleich weit → Punkt liegt nicht direkt
                auf einem Trainingspunkt → kein Memorization-Risiko.
    NNDR → 0  → Punkt ist fast identisch mit einem echten Trainingspunkt.

    Hinweis SMOTE: SMOTE interpoliert *zwischen* echten Punkten,
    daher liegt der nächste Nachbar typischerweise auf 0 Distanz.
    Das ist kein Bug, sondern zeigt, dass SMOTE keine echte Privatheit bietet.
    """
    rng = np.random.default_rng(seed)
    if len(synth) > sample_n:
        idx = rng.choice(len(synth), sample_n, replace=False)
        synth = synth[idx]

    nn = NearestNeighbors(
        n_neighbors=2,
        algorithm="brute",
        metric="euclidean",
        n_jobs=1
    ).fit(real_train) 
    
    dists, _ = nn.kneighbors(synth)          # shape (n, 2)
    eps = 1e-9
    nndr = dists[:, 0] / (dists[:, 1] + eps)
    median_nndr = round(float(np.median(nndr)), 4)

    pct_copies = round(float(np.mean(nndr < 0.05)) * 100, 1)
    return {"Median NNDR": median_nndr, "% quasi-Kopien": pct_copies}
