import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def generate(X_raw, y_raw, train_index, X_train_cols, seed, **kwargs):
    """
    SMOTE-Übersampling der Minderheitsklasse.

    Intern wird One-hot-Encoding auf den Trainingssubset angewendet und dann
    auf X_train_cols reindiziert, damit die Spaltenmenge identisch mit dem
    Klassifikator-Input ist — auch wenn seltene Kategorien nur im Test-Set vorkommen.
    """
    le = kwargs["le"]
    k_neighbors = kwargs.get("k_neighbors", 5)
    sampling_strategy = kwargs.get("sampling_strategy", "auto")

    X_tr = (
        pd.get_dummies(X_raw.loc[train_index])
        .reindex(columns=X_train_cols, fill_value=0)
    )
    y_tr = le.transform(y_raw.loc[train_index].values)

    print(
        f"   Erzeuge SMOTE-Samples: k_neighbors={k_neighbors}, "
        f"sampling_strategy={sampling_strategy}"
    )
    smote = SMOTE(
        random_state=seed,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
    )
    X_smote_arr, y_smote = smote.fit_resample(X_tr, y_tr)

    X_synth = pd.DataFrame(X_smote_arr, columns=X_train_cols)
    return X_synth, y_smote
