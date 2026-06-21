"""
=============================================================
  Synthetic Data Demo — Evaluation Script
  Dataset : UCI Adult Income
  Methods : Baseline (real) | SMOTE | CTGAN (SDV)
  Metrics : Utility · Fidelity · Privacy
=============================================================

Ausführung
----------
    python3 src/main.py [--no-ctgan] [--epochs N] [--seeds S [S ...]]

Flags
-----
    --no-ctgan     CTGAN überspringen (schneller, für Entwicklung)
    --epochs N     CTGAN-Epochen (default: 50)
    --seeds S ...  Zufalls-Seeds (default: 42)
"""

import argparse
import csv
import os
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

from metrics import utility_metrics, fidelity_metrics, privacy_nndr
from generators import REGISTRY

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-ctgan", action="store_true", help="CTGAN überspringen")
parser.add_argument("--epochs",   type=int, default=50,  help="CTGAN-Epochen")
parser.add_argument("--seeds",    nargs="+", type=int, default=[42],
                    help="Zufalls-Seeds (default: 42)")
args = parser.parse_args()

# ── Run timestamp (shared across plots and log) ────────────────────────────────
_RUN_DT     = datetime.now()
TS_DISPLAY  = _RUN_DT.strftime("%Y-%m-%d %H:%M:%S")
TS_FILENAME = _RUN_DT.strftime("%Y-%m-%d_%H-%M-%S")

# ── Pfade ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "adult", "adult.data")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "results")
CSV_PATH   = os.path.join(OUT_DIR, "results.csv")

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

num_cols_enc = [c for c in NUMERIC_COLS if c in X_enc.columns]

# ══════════════════════════════════════════════════════════════════════════════
# 2 · HILFSFUNKTIONEN — LOG-INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

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


# ── CSV-Header (einmalig prüfen, bevor die Seed-Schleife startet) ──────────────
_CSV_FIELDS = [
    "timestamp", "dataset", "model", "seed",
    "f1_macro", "recall_minority", "ks_mean", "median_nndr", "pct_copies",
]
_csv_is_new = not os.path.exists(CSV_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# 3 · SEED-SCHLEIFE
# ══════════════════════════════════════════════════════════════════════════════
_TOTAL_SECTIONS = 1 + len(REGISTRY)   # Baseline + alle Generatoren

for seed in args.seeds:

    if len(args.seeds) > 1:
        print(f"\n{'━' * 62}")
        print(f"  SEED {seed}")
        print(f"{'━' * 62}")

    # ── Train/Test-Split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train_df = pd.DataFrame(X_train.values, columns=X_train.columns)

    print(f"   Trainingsgröße  : {len(X_train):,} Zeilen")
    print(f"   Testgröße       : {len(X_test):,}  Zeilen")
    print(f"   Klassenverteilung (Train):")
    vc = pd.Series(y_train).value_counts()
    for cls, cnt in vc.items():
        label = le.inverse_transform([cls])[0]
        print(f"      {label:<8}  {cnt:,}  ({cnt/len(y_train)*100:.1f} %)")

    # ══════════════════════════════════════════════════════════════════════════
    # 3a · BASELINE — reale Trainingsdaten
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 62)
    print(f"  [1/{_TOTAL_SECTIONS}] BASELINE  (echte Trainingsdaten)")
    print("─" * 62)

    clf_base = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)

    util_base     = utility_metrics(y_test, y_pred_base)
    fidelity_base = {"Ø KS-Statistik": 0.0}
    privacy_base  = privacy_nndr(X_train.values, X_train.values, seed=seed)

    print("   Utility  :", util_base)
    print("   Fidelity :", fidelity_base, "  (Referenz)")
    print("   Privacy  :", privacy_base,  "  (Selbst-Distanz — Untergrenze)")

    results = {
        "Baseline": {
            "utility":  util_base,
            "fidelity": fidelity_base,
            "privacy":  privacy_base,
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 3b · GENERATOR-SCHLEIFE  (SMOTE, CTGAN, …)
    # ══════════════════════════════════════════════════════════════════════════
    _section_titles = {
        "SMOTE": "SMOTE  (interpolierte synthetische Samples)",
        "CTGAN": f"CTGAN  (SDV · {args.epochs} Epochen)",
    }

    for i, (name, gen_fn) in enumerate(REGISTRY.items(), 2):
        print("\n" + "─" * 62)

        if name == "CTGAN" and args.no_ctgan:
            print(f"  [{i}/{_TOTAL_SECTIONS}] CTGAN  → übersprungen (--no-ctgan)")
            print("─" * 62)
            continue

        print(f"  [{i}/{_TOTAL_SECTIONS}] {_section_titles.get(name, name)}")
        print("─" * 62)

        # Modellspezifische kwargs zusammenstellen
        gen_kwargs = {"le": le}
        if name == "CTGAN":
            gen_kwargs["epochs"] = args.epochs

        # Generator aufrufen (CTGAN mit Fehlerbehandlung wie bisher)
        if name == "CTGAN":
            try:
                X_synth, y_synth = gen_fn(
                    X_raw, y_raw, X_train.index, X_train.columns, seed, **gen_kwargs
                )
            except Exception as e:
                print(f"   ⚠  CTGAN fehlgeschlagen: {e}")
                continue
        else:
            X_synth, y_synth = gen_fn(
                X_raw, y_raw, X_train.index, X_train.columns, seed, **gen_kwargs
            )

        # SMOTE-spezifische Ausgabe: Klassenverteilung nach Resampling
        if name == "SMOTE":
            vc_after = pd.Series(y_synth).value_counts()
            print(f"   Trainingsgröße nach SMOTE: {len(X_synth):,}")
            for cls, cnt in vc_after.items():
                label = le.inverse_transform([cls])[0]
                print(f"      {label:<8}  {cnt:,}  ({cnt/len(y_synth)*100:.1f} %)")

        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        clf.fit(X_synth, y_synth)
        y_pred = clf.predict(X_test)

        util  = utility_metrics(y_test, y_pred)
        fidel = fidelity_metrics(X_train_df[num_cols_enc], X_synth[num_cols_enc], num_cols_enc)
        priv  = privacy_nndr(X_train.values, X_synth.values, seed=seed)

        print("   Utility  :", util)
        print("   Fidelity :", fidel)
        print("   Privacy  :", priv)

        results[name] = {"utility": util, "fidelity": fidel, "privacy": priv}

    # ══════════════════════════════════════════════════════════════════════════
    # 3c · ZUSAMMENFASSUNG
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # 3d · PLOTS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n▶  Erstelle Plots …")

    methods = list(results.keys())
    colors  = ["#4C72B0", "#DD8452", "#55A868"][: len(methods)]

    f1_vals  = [results[m]["utility"]["F1 (macro)"]      for m in methods]
    rec_vals = [results[m]["utility"]["Recall >50K"]     for m in methods]
    ks_vals  = [results[m]["fidelity"]["Ø KS-Statistik"] for m in methods]
    nn_vals  = [results[m]["privacy"]["Median NNDR"]     for m in methods]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Synthetic Data Evaluation  ·  UCI Adult Income",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 7a  F1 ────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(methods, f1_vals, color=colors, edgecolor="white", width=0.5)
    ax1.set_title("Utility — F1 Score (macro)", fontsize=11)
    ax1.set_ylabel("F1")
    ax1.set_ylim(0, 1)
    ax1.axhline(f1_vals[0], color=colors[0], linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, f1_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # ── 7b  Recall ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(methods, rec_vals, color=colors, edgecolor="white", width=0.5)
    ax2.set_title("Utility — Recall Minderheitsklasse (>50K)", fontsize=11)
    ax2.set_ylabel("Recall")
    ax2.set_ylim(0, 1)
    ax2.axhline(rec_vals[0], color=colors[0], linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, rec_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # ── 7c  KS-Statistik ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(methods, ks_vals, color=colors, edgecolor="white", width=0.5)
    ax3.set_title("Fidelity — Ø KS-Statistik (↓ besser)", fontsize=11)
    ax3.set_ylabel("KS")
    ax3.set_ylim(0, max(ks_vals) * 1.3 + 0.02)
    for bar, val in zip(bars, ks_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # ── 7d  NNDR ──────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(methods, nn_vals, color=colors, edgecolor="white", width=0.5)
    ax4.set_title("Privacy — Median NNDR (↑ besser)", fontsize=11)
    ax4.set_ylabel("NNDR")
    ax4.set_ylim(0, max(nn_vals) * 1.3 + 0.02)
    for bar, val in zip(bars, nn_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    _param_str = (
        f"Run: {TS_DISPLAY}  |  seed={seed}  |  epochs={args.epochs}"
        + ("  |  CTGAN: skipped" if args.no_ctgan else "  |  CTGAN: yes")
    )
    fig.text(0.5, 0.005, _param_str, ha="center", va="bottom",
             fontsize=8, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    plot_path = os.path.join(OUT_DIR, f"evaluation_results_{TS_FILENAME}_seed{seed}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot gespeichert: {plot_path}")

    # ── SMOTE-Verteilungsplot ──────────────────────────────────────────────────
    if "SMOTE" in results:
        X_smote_ref, y_smote_ref = (
            REGISTRY["SMOTE"](X_raw, y_raw, X_train.index, X_train.columns, seed, le=le)
        )

        fig2, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig2.suptitle(f"SMOTE — Klassenverteilung  ·  {TS_DISPLAY}  ·  seed={seed}",
                      fontsize=12, fontweight="bold")

        labels_  = [le.inverse_transform([0])[0], le.inverse_transform([1])[0]]
        before_  = [np.sum(y_train == 0), np.sum(y_train == 1)]
        after_   = [np.sum(y_smote_ref == 0), np.sum(y_smote_ref == 1)]

        axes[0].bar(labels_, before_, color=["#4C72B0", "#DD8452"], edgecolor="white")
        axes[0].set_title("Vor SMOTE")
        axes[0].set_ylabel("Anzahl Samples")
        for j, v in enumerate(before_):
            axes[0].text(j, v + 20, str(v), ha="center", fontsize=10)

        axes[1].bar(labels_, after_, color=["#4C72B0", "#DD8452"], edgecolor="white")
        axes[1].set_title("Nach SMOTE")
        axes[1].set_ylabel("Anzahl Samples")
        for j, v in enumerate(after_):
            axes[1].text(j, v + 20, str(v), ha="center", fontsize=10)

        plt.tight_layout()
        dist_path = os.path.join(OUT_DIR, f"smote_distribution_{TS_FILENAME}_seed{seed}.png")
        plt.savefig(dist_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Plot gespeichert: {dist_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3e · EXPERIMENT LOG
    # ══════════════════════════════════════════════════════════════════════════
    LOG_PATH   = os.path.join(SCRIPT_DIR, "..", "results", "experiment_log.md")
    is_new_log = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a") as log:
        if is_new_log:
            log.write("# Experiment Log — Synthetic Data Evaluation\n\n")
            log.write("UCI Adult Income dataset. Each run appends one block below.\n\n")
            log.write("---\n\n")
            log.write(METRIC_LEGEND)
            log.write("---\n\n")

        ctgan_ran = (not args.no_ctgan) and "CTGAN" in results
        log.write(f"## Run — {TS_DISPLAY}\n\n")
        log.write(
            f"**Parameters:** seed={seed}, epochs={args.epochs}, "
            f"CTGAN={'yes' if ctgan_ran else 'skipped (--no-ctgan)'}\n\n"
        )

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

        log.write("\n### Interpretation\n\n")
        baseline_f1  = results["Baseline"]["utility"]["F1 (macro)"]
        baseline_rec = results["Baseline"]["utility"]["Recall >50K"]
        for method, r in results.items():
            log.write(interpret_method(method, r, baseline_f1, baseline_rec))
            log.write("\n\n")

        log.write("---\n\n")

    print(f"▶  Experiment log updated: {os.path.abspath(LOG_PATH)}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3f · SAMMEL-CSV  (append, eine Zeile pro Methode+Seed)
    # ══════════════════════════════════════════════════════════════════════════
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_FIELDS)
        if _csv_is_new:
            writer.writeheader()
            _csv_is_new = False
        for method, r in results.items():
            writer.writerow({
                "timestamp":       TS_DISPLAY,
                "dataset":         "uci-adult",
                "model":           method,
                "seed":            seed,
                "f1_macro":        r["utility"]["F1 (macro)"],
                "recall_minority": r["utility"]["Recall >50K"],
                "ks_mean":         r["fidelity"]["Ø KS-Statistik"],
                "median_nndr":     r["privacy"]["Median NNDR"],
                "pct_copies":      r["privacy"]["% quasi-Kopien"],
            })

    print(f"▶  CSV aktualisiert: {os.path.abspath(CSV_PATH)}")

print("\n" + "═" * 62)
print("  Fertig.")
print("═" * 62 + "\n")
