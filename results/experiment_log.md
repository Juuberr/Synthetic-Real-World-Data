# Experiment Log — Synthetic Data Evaluation

UCI Adult Income dataset. Each run appends one block below.

---

## Metric Definitions

| Metric | Axis | Direction | What it measures |
|---|---|---|---|
| **F1 (macro)** | Utility | ↑ higher is better | Harmonic mean of precision & recall, averaged equally over both classes. Penalises ignoring either class. |
| **Recall >50K** | Utility | ↑ higher is better | Fraction of true high-earners the classifier correctly identifies. Key for the minority class. |
| **Ø KS-Statistic** | Fidelity | ↓ lower is better | Average Kolmogorov–Smirnov distance between synthetic and real numeric distributions. 0 = identical, 1 = no overlap. |
| **Median NNDR** | Privacy | ↑ higher is better | Nearest-Neighbour Distance Ratio: dist(1st real neighbour) / dist(2nd real neighbour). Near 1 → synthetic point is not memorising any single real record. Near 0 → near-copy of a training point. |
| **% quasi-copies** | Privacy | ↓ lower is better | Percentage of synthetic points with NNDR < 0.05 — effectively identical to a real training row. |

---

## Run — 2026-06-19 13:33:37

**Parameters:** epochs=50, CTGAN=yes

### Results

| Method | F1 (macro) | Recall >50K | Ø KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 100.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 66.6% |
| CTGAN | 0.4289 | 0.0000 | 0.2687 | 0.9734 | 0.4% |

### Interpretation

**Baseline**
- Utility: reference point — trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance — trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** — Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** — KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** — Median NNDR = 0.0000, 66.6% quasi-copies. Synthetic points sit directly on top of real training records.

**CTGAN**
- Utility: **complete failure** — Recall >50K = 0.000. The classifier trained on this synthetic data predicts every sample as ≤50K. F1 (0.4289, -0.3579 vs baseline) is driven purely by the majority class.
- Fidelity: **poor** — KS = 0.2687. Large distributional gap; synthetic numerics differ substantially from real data.
- Privacy: **strong** — Median NNDR = 0.9734, 0.4% quasi-copies. Synthetic points are genuinely distant from real training records.

---

## Run — 2026-06-19 14:31:53

**Parameters:** epochs=150, CTGAN=yes

### Results

| Method | F1 (macro) | Recall >50K | Ø KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 100.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 66.6% |
| CTGAN | 0.4289 | 0.0000 | 0.3028 | 0.9720 | 0.8% |

### Interpretation

**Baseline**
- Utility: reference point — trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance — trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** — Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** — KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** — Median NNDR = 0.0000, 66.6% quasi-copies. Synthetic points sit directly on top of real training records.

**CTGAN**
- Utility: **complete failure** — Recall >50K = 0.000. The classifier trained on this synthetic data predicts every sample as ≤50K. F1 (0.4289, -0.3579 vs baseline) is driven purely by the majority class.
- Fidelity: **poor** — KS = 0.3028. Large distributional gap; synthetic numerics differ substantially from real data.
- Privacy: **strong** — Median NNDR = 0.9720, 0.8% quasi-copies. Synthetic points are genuinely distant from real training records.

---

## Run — 2026-06-19 14:53:06

**Parameters:** epochs=50, CTGAN=yes

### Results

| Method | F1 (macro) | Recall >50K | Ø KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 100.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 66.6% |
| CTGAN | 0.7387 | 0.5153 | 0.1938 | 0.8371 | 0.1% |

### Interpretation

**Baseline**
- Utility: reference point — trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance — trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** — Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** — KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** — Median NNDR = 0.0000, 66.6% quasi-copies. Synthetic points sit directly on top of real training records.

**CTGAN**
- Utility: **below baseline** — Recall >50K 0.5153 (-0.1092), F1 0.7387 (-0.0481). Synthetic data under-represents the minority class.
- Fidelity: **moderate** — KS = 0.1938. Noticeable drift; the generator did not fully capture real distributions.
- Privacy: **strong** — Median NNDR = 0.8371, 0.1% quasi-copies. Synthetic points are genuinely distant from real training records.

---

## Run — 2026-06-19 15:02:57

**Parameters:** epochs=150, CTGAN=yes

### Results

| Method | F1 (macro) | Recall >50K | Ø KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 100.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 66.6% |
| CTGAN | 0.7440 | 0.5273 | 0.2056 | 0.8604 | 0.1% |

### Interpretation

**Baseline**
- Utility: reference point — trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance — trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** — Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** — KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** — Median NNDR = 0.0000, 66.6% quasi-copies. Synthetic points sit directly on top of real training records.

**CTGAN**
- Utility: **below baseline** — Recall >50K 0.5273 (-0.0972), F1 0.7440 (-0.0428). Synthetic data under-represents the minority class.
- Fidelity: **moderate** — KS = 0.2056. Noticeable drift; the generator did not fully capture real distributions.
- Privacy: **strong** — Median NNDR = 0.8604, 0.1% quasi-copies. Synthetic points are genuinely distant from real training records.

---

## Run — 2026-06-19 15:53:04

**Parameters:** epochs=50, CTGAN=yes

### Results

| Method | F1 (macro) | Recall >50K | Ø KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 100.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 66.6% |
| CTGAN | 0.7083 | 0.4035 | 0.1854 | 0.8389 | 0.1% |

### Interpretation

**Baseline**
- Utility: reference point — trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance — trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** — Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** — KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** — Median NNDR = 0.0000, 66.6% quasi-copies. Synthetic points sit directly on top of real training records.

**CTGAN**
- Utility: **below baseline** — Recall >50K 0.4035 (-0.2210), F1 0.7083 (-0.0785). Synthetic data under-represents the minority class.
- Fidelity: **moderate** — KS = 0.1854. Noticeable drift; the generator did not fully capture real distributions.
- Privacy: **strong** — Median NNDR = 0.8389, 0.1% quasi-copies. Synthetic points are genuinely distant from real training records.

---

## Run � 2026-06-22 09:12:45

**Parameters:** seed=42, epochs=50, CTGAN=skipped (--no-ctgan)

### Results

| Method | F1 (macro) | Recall >50K | � KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 0.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 0.0% |

### Interpretation

**Baseline**
- Utility: reference point � trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance � trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** � Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** � KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** � Median NNDR = 0.0000, 0.0% quasi-copies. Synthetic points sit directly on top of real training records.

---

## Run � 2026-06-22 09:23:28

**Parameters:** seed=42, epochs=50, CTGAN=skipped (--no-ctgan)

### Results

| Method | F1 (macro) | Recall >50K | � KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 0.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 0.0% |

### Interpretation

**Baseline**
- Utility: reference point � trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance � trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** � Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** � KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** � Median NNDR = 0.0000, 0.0% quasi-copies. Synthetic points sit directly on top of real training records.

---

## Run � 2026-06-22 09:24:02

**Parameters:** seed=42, epochs=50, CTGAN=skipped (--no-ctgan)

### Results

| Method | F1 (macro) | Recall >50K | � KS | Median NNDR | % quasi-copies |
|---|---|---|---|---|---|
| Baseline | 0.7868 | 0.6245 | 0.0000 | 0.0000 | 0.0% |
| SMOTE | 0.7880 | 0.6485 | 0.0629 | 0.0000 | 0.0% |
| MST | 0.6824 | 0.4241 | 0.4112 | 0.0000 | 0.0% |

### Interpretation

**Baseline**
- Utility: reference point � trained and tested on real data.
- Fidelity: KS = 0.000 by definition (real vs. real).
- Privacy: NNDR = 0.000 (self-distance � trivial lower bound, not a real privacy measure).

**SMOTE**
- Utility: **matches baseline** � Recall >50K 0.6485 (+0.0240), F1 0.7880 (+0.0012). Synthetic data preserves enough signal for downstream use.
- Fidelity: **good** � KS = 0.0629. Minor distributional drift in some numeric columns.
- Privacy: **no privacy** � Median NNDR = 0.0000, 0.0% quasi-copies. Synthetic points sit directly on top of real training records.

**MST**
- Utility: **below baseline** � Recall >50K 0.4241 (-0.2004), F1 0.6824 (-0.1044). Synthetic data under-represents the minority class.
- Fidelity: **poor** � KS = 0.4112. Large distributional gap; synthetic numerics differ substantially from real data.
- Privacy: **no privacy** � Median NNDR = 0.0000, 0.0% quasi-copies. Synthetic points sit directly on top of real training records.

---

