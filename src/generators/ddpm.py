"""
TabDDPM-Generator über die synthcity-Bibliothek (Plugin "ddpm").

NICHT das Original-Repo yandex-research/tab-ddpm, sondern synthcity 0.2.12.
Verifizierter Stack: synthcity 0.2.12, torch 2.2.2, opacus 1.4.1, numpy 1.26.4
(conda-Env synth-eval, Python 3.11).

TabDDPM-spezifische Schritte
────────────────────────────
1. ROHE Kategorialspalten füttern — KEINE One-Hot-Dummies. (One-Hot war der
   CTGAN-Bug: SDV/Generator behandelt Binärspalten als kontinuierlich → Recall→0.)
2. Target-Spalte (income) über GenericDataLoader(target_column=...) als Label
   übergeben und Plugin mit is_classification=True instanziieren. Dadurch
   konditioniert das Diffusionsmodell auf die Klasse und erzeugt BEIDE
   income-Klassen (die Minderheit >50K geht nicht verloren).
3. device MUSS ein torch.device-Objekt sein (String 'cpu' wirft pydantic-Fehler).
4. seed → random_state des Plugins (passend zur --seeds-Logik in main.py).
5. Output zurück ins One-Hot-Format der Klassifikator-Spalten:
   pd.get_dummies(...).reindex(columns=X_train_cols, fill_value=0).

WICHTIG — OpenMP-Deadlock
─────────────────────────
TabDDPM trainiert auf CPU mit torch. Auf macOS verklemmen sich mehrere
OpenMP-Laufzeiten (torch + KeOps) an einer Fork-Barriere → fit() hängt bei
0% CPU und kommt nie zurück. Gegenmaßnahmen:
  - main.py setzt OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE vor dem
    ersten torch-Import (das ist der eigentliche Fix).
  - torch.set_num_threads(1) unten als zusätzliche Absicherung.
Single-Thread ist hier kein Performance-Problem: ~0.27s/Epoche auf UCI Adult.
"""

import torch
import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# Absicherung gegen den OpenMP-Fork-Barrier-Deadlock (siehe Docstring oben).
torch.set_num_threads(1)


def generate(X_raw, y_raw, train_index, X_train_cols, seed, **kwargs):
    """
    TabDDPM-Synthesizer (synthcity, Plugin "ddpm").

    Bekommt rohe Kategorialspalten — kein One-hot-Encoding als Eingabe.
    One-hot wird erst auf dem Output angewendet und auf X_train_cols reindiziert.

    kwargs
    ------
    le          sklearn LabelEncoder (Pflicht) — kodiert income → 0/1
    n_iter      Trainingsiterationen des Diffusionsmodells (default 300)
    target_col  Zielspaltenname (default "income")
    batch_size  Batch-Größe (default 1024)
    num_timesteps  Diffusions-Zeitschritte (default 100 — bestimmt v.a. die
                   Sampling-Dauer; 100 reicht hier für realistische Klassenbalance)
    """
    le            = kwargs["le"]
    n_iter        = kwargs.get("n_iter", 300)
    target_col    = kwargs.get("target_col", "income")
    batch_size    = kwargs.get("batch_size", 1024)
    num_timesteps = kwargs.get("num_timesteps", 100)

    # Rohe Feature- + Zielspalten nur für den Trainings-Split zusammenbauen.
    train_data = X_raw.loc[train_index].copy()
    train_data[target_col] = y_raw.loc[train_index].values
    train_data = train_data.reset_index(drop=True)

    # Target als Label an den DataLoader geben → Conditioning auf die Klasse.
    loader = GenericDataLoader(train_data, target_column=target_col)

    print(f"   Trainiere TabDDPM ({n_iter} Iterationen) …")
    plugin = Plugins().get(
        "ddpm",
        is_classification=True,
        n_iter=n_iter,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        device=torch.device("cpu"),
        random_state=seed,
    )
    plugin.fit(loader)

    synthetic = plugin.generate(count=len(train_data)).dataframe()

    y_synth     = le.transform(synthetic[target_col].astype(str).str.strip())
    X_synth_raw = synthetic.drop(target_col, axis=1)
    X_synth     = pd.get_dummies(X_synth_raw).reindex(columns=X_train_cols, fill_value=0)

    return X_synth, y_synth
