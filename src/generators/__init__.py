"""
Hinzufügen eines neuen Generators
──────────────────────────────────
1. Neue Datei  src/generators/<name>.py  anlegen mit:

       def generate(X_raw, y_raw, train_index, X_train_cols, seed, **kwargs):
           ...
           return X_synth, y_synth

   Erwartetes Format
   -----------------
   Eingabe:
     X_raw        pd.DataFrame  – rohe Feature-Spalten, NICHT one-hot-kodiert
     y_raw        pd.Series     – Rohziel-Labels (Strings, z.B. "<=50K")
     train_index  pd.Index      – Zeilenauswahl für den Trainings-Split
     X_train_cols pd.Index      – One-hot-Spaltennamen des Klassifikators
     seed         int           – Zufalls-Seed
     **kwargs                   – mindestens le (sklearn LabelEncoder);
                                  weitere modellspezifische Parameter nach Bedarf

   Ausgabe:
     X_synth  pd.DataFrame  – synthetische Features im One-hot-Format von X_train_cols
     y_synth  np.ndarray    – integer-kodierte Ziel-Labels (0/1)

   WICHTIG: Dem Generator IMMER rohe Kategorialspalten (X_raw) übergeben.
   One-hot-Kodierung erst intern anwenden (wie in smote.py) oder gar nicht
   (wie in ctgan.py, wo SDV selbst mit Rohdaten umgeht). Wird stattdessen
   X_enc direkt übergeben, behandelt CTGAN Binärspalten als kontinuierliche
   Werte → Recall kollabiert auf 0.

   Am Ende immer  .reindex(columns=X_train_cols, fill_value=0)  aufrufen,
   damit die Spaltenmenge garantiert mit dem Klassifikator übereinstimmt.

2. Eintrag in REGISTRY unten ergänzen:

       from generators.<name> import generate as <name>_generate
       REGISTRY["<AnzeigeName>"] = <name>_generate

3. Falls das Modell eigene kwargs braucht (z.B. epochs): in main.py im
   gen_kwargs-Block für den entsprechenden Namen ergänzen.
"""

from generators.smote import generate as smote_generate
from generators.ctgan import generate as ctgan_generate
from generators.privbayes import generate as privbayes_generate

REGISTRY = {
    "SMOTE": smote_generate,
    "CTGAN": ctgan_generate,
    "PrivBayes": privbayes_generate,
}
