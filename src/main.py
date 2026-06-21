"""
=============================================================
  Synthetic Data Demo — Evaluation Script
  Dataset : UCI Adult Income
  Methods : Baseline (real) | SMOTE | CTGAN (SDV)
  Metrics : Utility · Fidelity · Privacy
=============================================================

Ausführung
----------
    cd test/Synthetic-Real-World-Data-main
    python ../demo/evaluate.py

Flags
-----
    --no-ctgan   CTGAN überspringen (schneller, für Entwicklung)
    --epochs N   CTGAN-Epochen (default: 50)
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ks_2samp

from imblearn.over_sampling import SMOTE

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-ctgan", action="store_true", help="CTGAN überspringen")
parser.add_argument("--epochs",   type=int, default=50,  help="CTGAN-Epochen")
args = parser.parse_args()

# ── Run timestamp (shared across plots and log) ────────────────────────────────
_RUN_DT     = datetime.now()
TS_DISPLAY  = _RUN_DT.strftime("%Y-%m-%d %H:%M:%S")
TS_FILENAME = _RUN_DT.strftime("%Y-%m-%d_%H-%M-%S")

# ── Pfade ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "adult", "adult.data")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "results")

# ══════════════════════════════════════════════════════════════════════════════
# 1 · DATEN LADEN & VORBEREITEN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("  SYNTHETIC DATA DEMO  ·  UCI Adult Income")
print("═" * 62)

print("\n▶  Lade Daten …")
cols = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
df = pd.read_csv(DATA_PATH, header=None, sep=",", skipinitialspace=True, names=cols)
df = df.replace("?", pd.NA).dropna()
df["income"] = df["income"].str.strip()

NUMERIC_COLS = ["age", "fnlwgt", "education_num", "capital_gain",
                "capital_loss", "hours_per_week"]

y_raw = df["income"]
X_raw = df.drop("income", axis=1)
X_enc = pd.get_dummies(X_raw)

le = LabelEncoder()
y  = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y
)

# numerische Spalten im encodierten DataFrame
num_cols_enc = [c for c in NUMERIC_COLS if c in X_enc.columns]

print(f"   Trainingsgröße  : {len(X_train):,} Zeilen")
print(f"   Testgröße       : {len(X_test):,}  Zeilen")
print(f"   Klassenverteilung (Train):")
vc = pd.Series(y_train).value_counts()
for cls, cnt in vc.items():
    label = le.inverse_transform([cls])[0]
    print(f"      {label:<8}  {cnt:,}  ({cnt/len(y_train)*100:.1f} %)")

# ══════════════════════════════════════════════════════════════════════════════
# 2 · HILFSFUNKTIONEN — METRIKEN
# ══════════════════════════════════════════════════════════════════════════════

def utility_metrics(y_true, y_pred, label=""):
    """F1 (macro) und Recall der Minderheitsklasse (class=1, d.h. >50K)."""
    f1  = f1_score(y_true, y_pred, average="macro",     zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1,     zero_division=0)
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


def privacy_nndr(real_train: np.ndarray, synth: np.ndarray, sample_n=2000):
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
    rng = np.random.default_rng(42)
    if len(synth) > sample_n:
        idx = rng.choice(len(synth), sample_n, replace=False)
        synth = synth[idx]

    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(real_train)
    dists, _ = nn.kneighbors(synth)          # shape (n, 2)
    eps = 1e-9
    nndr = dists[:, 0] / (dists[:, 1] + eps)
    median_nndr = round(float(np.median(nndr)), 4)

    # Anteil synthetischer Punkte mit NNDR < 0.05 (quasi-Kopien)
    pct_copies = round(float(np.mean(nndr < 0.05)) * 100, 1)
    return {"Median NNDR": median_nndr, "% quasi-Kopien": pct_copies}


# ══════════════════════════════════════════════════════════════════════════════
# 3 · BASELINE — reale Trainingsdaten
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  [1/3] BASELINE  (echte Trainingsdaten)")
print("─" * 62)

clf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_base.fit(X_train, y_train)
y_pred_base = clf_base.predict(X_test)

util_base     = utility_metrics(y_test, y_pred_base)
fidelity_base = {"Ø KS-Statistik": 0.0}   # real vs. real = 0 per Definition
privacy_base  = privacy_nndr(X_train.values, X_train.values)

print("   Utility  :", util_base)
print("   Fidelity :", fidelity_base, "  (Referenz)")
print("   Privacy  :", privacy_base,  "  (Selbst-Distanz — Untergrenze)")

# ══════════════════════════════════════════════════════════════════════════════
# 4 · SMOTE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  [2/3] SMOTE  (interpolierte synthetische Samples)")
print("─" * 62)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

vc_after = pd.Series(y_smote).value_counts()
print(f"   Trainingsgröße nach SMOTE: {len(X_smote):,}")
for cls, cnt in vc_after.items():
    label = le.inverse_transform([cls])[0]
    print(f"      {label:<8}  {cnt:,}  ({cnt/len(y_smote)*100:.1f} %)")

clf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_smote.fit(X_smote, y_smote)
y_pred_smote = clf_smote.predict(X_test)

# für Fidelity: SMOTE-Punkte sind alle numerischen Features
X_smote_df = pd.DataFrame(X_smote, columns=X_train.columns)
X_train_df  = pd.DataFrame(X_train.values, columns=X_train.columns)

util_smote     = utility_metrics(y_test, y_pred_smote)
fidelity_smote = fidelity_metrics(X_train_df[num_cols_enc],
                                   X_smote_df[num_cols_enc],
                                   num_cols_enc)
privacy_smote  = privacy_nndr(X_train.values, X_smote.values)

print("   Utility  :", util_smote)
print("   Fidelity :", fidelity_smote)
print("   Privacy  :", privacy_smote)

# ══════════════════════════════════════════════════════════════════════════════
# 5 · CTGAN (SDV)
# ══════════════════════════════════════════════════════════════════════════════
results = {
    "Baseline": {"utility": util_base, "fidelity": fidelity_base, "privacy": privacy_base},
    "SMOTE":    {"utility": util_smote, "fidelity": fidelity_smote, "privacy": privacy_smote},
}

if args.no_ctgan:
    print("\n" + "─" * 62)
    print("  [3/3] CTGAN  → übersprungen (--no-ctgan)")
    print("─" * 62)
    util_ctgan = fidelity_ctgan = privacy_ctgan = None
else:
    print("\n" + "─" * 62)
    print(f"  [3/3] CTGAN  (SDV · {args.epochs} Epochen)")
    print("─" * 62)

    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import Metadata

        # Feed raw categoricals to SDV — CTGAN handles them natively via its
        # conditional generator; one-hot dummies break this and cause income
        # to be treated as a continuous float, making recall collapse to 0.
        train_data_sdv = X_raw.loc[X_train.index].copy()
        train_data_sdv["income"] = y_raw.loc[X_train.index].values

        metadata = Metadata.detect_from_dataframe(data=train_data_sdv)

        print(f"   Trainiere CTGAN ({args.epochs} Epochen) …")
        sdv_model = CTGANSynthesizer(metadata, epochs=args.epochs, verbose=False)
        sdv_model.fit(train_data_sdv)

        synthetic_sdv = sdv_model.sample(num_rows=len(train_data_sdv))
        y_synth_sdv   = le.transform(synthetic_sdv["income"].str.strip())
        X_synth_raw   = synthetic_sdv.drop("income", axis=1)
        X_synth_sdv   = pd.get_dummies(X_synth_raw).reindex(columns=X_train.columns, fill_value=0)

        clf_ctgan = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_ctgan.fit(X_synth_sdv, y_synth_sdv)
        y_pred_ctgan = clf_ctgan.predict(X_test)

        X_synth_df  = pd.DataFrame(X_synth_sdv.values, columns=X_train.columns)

        util_ctgan     = utility_metrics(y_test, y_pred_ctgan)
        fidelity_ctgan = fidelity_metrics(X_train_df[num_cols_enc],
                                           X_synth_df[num_cols_enc],
                                           num_cols_enc)
        privacy_ctgan  = privacy_nndr(X_train.values, X_synth_sdv.values)

        print("   Utility  :", util_ctgan)
        print("   Fidelity :", fidelity_ctgan)
        print("   Privacy  :", privacy_ctgan)

        results["CTGAN"] = {
            "utility":  util_ctgan,
            "fidelity": fidelity_ctgan,
            "privacy":  privacy_ctgan,
        }

    except Exception as e:
        print(f"   ⚠  CTGAN fehlgeschlagen: {e}")
        util_ctgan = fidelity_ctgan = privacy_ctgan = None

# ══════════════════════════════════════════════════════════════════════════════
# 6 · ZUSAMMENFASSUNG
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("  ERGEBNISÜBERSICHT")
print("═" * 62)
header = f"{'Methode':<12}  {'F1 (macro)':>11}  {'Recall >50K':>11}  {'Ø KS':>7}  {'NNDR':>7}  {'% Kopien':>9}"
print(header)
print("─" * len(header))

for method, r in results.items():
    f1_  = r["utility"]["F1 (macro)"]
    rec_ = r["utility"]["Recall >50K"]
    ks_  = r["fidelity"]["Ø KS-Statistik"]
    nn_  = r["privacy"]["Median NNDR"]
    cp_  = r["privacy"]["% quasi-Kopien"]
    print(f"{method:<12}  {f1_:>11.4f}  {rec_:>11.4f}  {ks_:>7.4f}  {nn_:>7.4f}  {cp_:>8.1f}%")

print()
print("  Utility  → F1 und Recall — höher ist besser")
print("  Fidelity → KS-Statistik — niedriger = ähnlicher zu echten Daten")
print("  Privacy  → NNDR         — höher = mehr Abstand zu echten Punkten")
print("             % Kopien     — Anteil Synth.-Punkte nahe echtem Datenpunkt")

# ══════════════════════════════════════════════════════════════════════════════
# 7 · PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶  Erstelle Plots …")

methods = list(results.keys())
colors  = ["#4C72B0", "#DD8452", "#55A868"][: len(methods)]

f1_vals  = [results[m]["utility"]["F1 (macro)"]    for m in methods]
rec_vals = [results[m]["utility"]["Recall >50K"]   for m in methods]
ks_vals  = [results[m]["fidelity"]["Ø KS-Statistik"] for m in methods]
nn_vals  = [results[m]["privacy"]["Median NNDR"]   for m in methods]

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Synthetic Data Evaluation  ·  UCI Adult Income",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 7a  F1 ────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(methods, f1_vals, color=colors, edgecolor="white", width=0.5)
ax1.set_title("Utility — F1 Score (macro)", fontsize=11)
ax1.set_ylabel("F1")
ax1.set_ylim(0, 1)
ax1.axhline(f1_vals[0], color=colors[0], linewidth=1, linestyle="--", alpha=0.5)
for bar, val in zip(bars, f1_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# ── 7b  Recall ────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(methods, rec_vals, color=colors, edgecolor="white", width=0.5)
ax2.set_title("Utility — Recall Minderheitsklasse (>50K)", fontsize=11)
ax2.set_ylabel("Recall")
ax2.set_ylim(0, 1)
ax2.axhline(rec_vals[0], color=colors[0], linewidth=1, linestyle="--", alpha=0.5)
for bar, val in zip(bars, rec_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# ── 7c  KS-Statistik ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(methods, ks_vals, color=colors, edgecolor="white", width=0.5)
ax3.set_title("Fidelity — Ø KS-Statistik (↓ besser)", fontsize=11)
ax3.set_ylabel("KS")
ax3.set_ylim(0, max(ks_vals) * 1.3 + 0.02)
for bar, val in zip(bars, ks_vals):
    ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# ── 7d  NNDR ──────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
bars = ax4.bar(methods, nn_vals, color=colors, edgecolor="white", width=0.5)
ax4.set_title("Privacy — Median NNDR (↑ besser)", fontsize=11)
ax4.set_ylabel("NNDR")
ax4.set_ylim(0, max(nn_vals) * 1.3 + 0.02)
for bar, val in zip(bars, nn_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)

_param_str = (
    f"Run: {TS_DISPLAY}  |  epochs={args.epochs}"
    + ("  |  CTGAN: skipped" if args.no_ctgan else "  |  CTGAN: yes")
)
fig.text(0.5, 0.005, _param_str, ha="center", va="bottom",
         fontsize=8, color="#555555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

plot_path = os.path.join(OUT_DIR, f"evaluation_results_{TS_FILENAME}.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Plot gespeichert: {plot_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 8 · KLASSENVERTEILUNG VOR / NACH SMOTE
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(9, 4))
fig2.suptitle(f"SMOTE — Klassenverteilung  ·  {TS_DISPLAY}", fontsize=12, fontweight="bold")

labels_  = [le.inverse_transform([0])[0], le.inverse_transform([1])[0]]
before_  = [np.sum(y_train == 0), np.sum(y_train == 1)]
after_   = [np.sum(y_smote == 0), np.sum(y_smote == 1)]

axes[0].bar(labels_, before_, color=["#4C72B0", "#DD8452"], edgecolor="white")
axes[0].set_title("Vor SMOTE")
axes[0].set_ylabel("Anzahl Samples")
for i, v in enumerate(before_):
    axes[0].text(i, v + 20, str(v), ha="center", fontsize=10)

axes[1].bar(labels_, after_, color=["#4C72B0", "#DD8452"], edgecolor="white")
axes[1].set_title("Nach SMOTE")
axes[1].set_ylabel("Anzahl Samples")
for i, v in enumerate(after_):
    axes[1].text(i, v + 20, str(v), ha="center", fontsize=10)

plt.tight_layout()
dist_path = os.path.join(OUT_DIR, f"smote_distribution_{TS_FILENAME}.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Plot gespeichert: {dist_path}")

print("\n" + "═" * 62)
print("  Fertig.")
print("═" * 62 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# 9 · EXPERIMENT LOG
# ══════════════════════════════════════════════════════════════════════════════

LOG_PATH = os.path.join(SCRIPT_DIR, "..", "results", "experiment_log.md")
is_new_file = not os.path.exists(LOG_PATH)

METRIC_LEGEND = """\
## Metric Definitions

| Metric | Axis | Direction | What it measures |
|---|---|---|---|
| **F1 (macro)** | Utility | ↑ higher is better | Harmonic mean of precision & recall, averaged equally over both classes. Penalises ignoring either class. |
| **Recall >50K** | Utility | ↑ higher is better | Fraction of true high-earners the classifier correctly identifies. Key for the minority class. |
| **Ø KS-Statistic** | Fidelity | ↓ lower is better | Average Kolmogorov–Smirnov distance between synthetic and real numeric distributions. 0 = identical, 1 = no overlap. |
| **Median NNDR** | Privacy | ↑ higher is better | Nearest-Neighbour Distance Ratio: dist(1st real neighbour) / dist(2nd real neighbour). Near 1 → synthetic point is not memorising any single real record. Near 0 → near-copy of a training point. |
| **% quasi-copies** | Privacy | ↓ lower is better | Percentage of synthetic points with NNDR < 0.05 — effectively identical to a real training row. |

"""


def interpret_method(name, r, baseline_f1, baseline_rec):
    f1  = r["utility"]["F1 (macro)"]
    rec = r["utility"]["Recall >50K"]
    ks  = r["fidelity"]["Ø KS-Statistik"]
    nn  = r["privacy"]["Median NNDR"]
    cp  = r["privacy"]["% quasi-Kopien"]

    lines = [f"**{name}**"]

    # Utility
    f1_delta  = f1  - baseline_f1
    rec_delta = rec - baseline_rec
    if name == "Baseline":
        lines.append("- Utility: reference point — trained and tested on real data.")
    else:
        f1_sign  = "+" if f1_delta  >= 0 else ""
        rec_sign = "+" if rec_delta >= 0 else ""
        if rec == 0.0:
            lines.append(
                f"- Utility: **complete failure** — Recall >50K = 0.000. The classifier "
                f"trained on this synthetic data predicts every sample as ≤50K. "
                f"F1 ({f1:.4f}, {f1_sign}{f1_delta:.4f} vs baseline) is driven purely by the majority class."
            )
        elif rec < baseline_rec - 0.05:
            lines.append(
                f"- Utility: **below baseline** — Recall >50K {rec:.4f} ({rec_sign}{rec_delta:.4f}), "
                f"F1 {f1:.4f} ({f1_sign}{f1_delta:.4f}). Synthetic data under-represents the minority class."
            )
        elif abs(rec_delta) <= 0.05 and abs(f1_delta) <= 0.02:
            lines.append(
                f"- Utility: **matches baseline** — Recall >50K {rec:.4f} ({rec_sign}{rec_delta:.4f}), "
                f"F1 {f1:.4f} ({f1_sign}{f1_delta:.4f}). Synthetic data preserves enough signal for downstream use."
            )
        else:
            lines.append(
                f"- Utility: Recall >50K {rec:.4f} ({rec_sign}{rec_delta:.4f}), "
                f"F1 {f1:.4f} ({f1_sign}{f1_delta:.4f})."
            )

    # Fidelity
    if name == "Baseline":
        lines.append("- Fidelity: KS = 0.000 by definition (real vs. real).")
    elif ks < 0.05:
        lines.append(f"- Fidelity: **excellent** — KS = {ks:.4f}. Numeric distributions are nearly indistinguishable from real data.")
    elif ks < 0.15:
        lines.append(f"- Fidelity: **good** — KS = {ks:.4f}. Minor distributional drift in some numeric columns.")
    elif ks < 0.25:
        lines.append(f"- Fidelity: **moderate** — KS = {ks:.4f}. Noticeable drift; the generator did not fully capture real distributions.")
    else:
        lines.append(f"- Fidelity: **poor** — KS = {ks:.4f}. Large distributional gap; synthetic numerics differ substantially from real data.")

    # Privacy
    if name == "Baseline":
        lines.append("- Privacy: NNDR = 0.000 (self-distance — trivial lower bound, not a real privacy measure).")
    elif nn < 0.05:
        lines.append(
            f"- Privacy: **no privacy** — Median NNDR = {nn:.4f}, {cp:.1f}% quasi-copies. "
            f"Synthetic points sit directly on top of real training records."
        )
    elif nn < 0.5:
        lines.append(
            f"- Privacy: **low** — Median NNDR = {nn:.4f}, {cp:.1f}% quasi-copies. "
            f"Many synthetic points are close to real training data."
        )
    elif nn < 0.8:
        lines.append(
            f"- Privacy: **moderate** — Median NNDR = {nn:.4f}, {cp:.1f}% quasi-copies. "
            f"Reasonable separation from training data."
        )
    else:
        lines.append(
            f"- Privacy: **strong** — Median NNDR = {nn:.4f}, {cp:.1f}% quasi-copies. "
            f"Synthetic points are genuinely distant from real training records."
        )

    return "\n".join(lines)


with open(LOG_PATH, "a") as log:
    if is_new_file:
        log.write("# Experiment Log — Synthetic Data Evaluation\n\n")
        log.write("UCI Adult Income dataset. Each run appends one block below.\n\n")
        log.write("---\n\n")
        log.write(METRIC_LEGEND)
        log.write("---\n\n")

    ts     = TS_DISPLAY
    epochs = getattr(args, "epochs", "N/A")
    ctgan_run = not getattr(args, "no_ctgan", False)

    log.write(f"## Run — {ts}\n\n")
    log.write(f"**Parameters:** epochs={epochs}, CTGAN={'yes' if ctgan_run else 'skipped (--no-ctgan)'}\n\n")

    # Results table
    log.write("### Results\n\n")
    log.write("| Method | F1 (macro) | Recall >50K | Ø KS | Median NNDR | % quasi-copies |\n")
    log.write("|---|---|---|---|---|---|\n")
    for method, r in results.items():
        f1_  = r["utility"]["F1 (macro)"]
        rec_ = r["utility"]["Recall >50K"]
        ks_  = r["fidelity"]["Ø KS-Statistik"]
        nn_  = r["privacy"]["Median NNDR"]
        cp_  = r["privacy"]["% quasi-Kopien"]
        log.write(f"| {method} | {f1_:.4f} | {rec_:.4f} | {ks_:.4f} | {nn_:.4f} | {cp_:.1f}% |\n")

    # Per-method interpretation
    log.write("\n### Interpretation\n\n")
    baseline_f1  = results["Baseline"]["utility"]["F1 (macro)"]
    baseline_rec = results["Baseline"]["utility"]["Recall >50K"]
    for method, r in results.items():
        log.write(interpret_method(method, r, baseline_f1, baseline_rec))
        log.write("\n\n")

    log.write("---\n\n")

print(f"▶  Experiment log updated: {os.path.abspath(LOG_PATH)}")
