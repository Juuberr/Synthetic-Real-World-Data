import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


def generate(X_raw, y_raw, train_index, X_train_cols, seed, **kwargs):
    """
    CTGAN-Synthesizer (SDV).

    Bekommt rohe Kategorialspalten — kein One-hot-Encoding als Eingabe.
    One-hot wird erst auf dem SDV-Output angewendet und auf X_train_cols
    reindiziert. Das verhindert den Recall-Kollaps, der entsteht, wenn
    SDV bereits kodierte Binärspalten als kontinuierliche Werte behandelt.

    Hardcoded: Ziel-Spaltenname "income" (UCI Adult). Für andere Datasets
    als target_col=... in kwargs übergeben.
    """
    le         = kwargs["le"]
    epochs     = kwargs.get("epochs", 50)
    target_col = kwargs.get("target_col", "income")

    train_data_sdv = X_raw.loc[train_index].copy()
    train_data_sdv[target_col] = y_raw.loc[train_index].values

    metadata = Metadata.detect_from_dataframe(data=train_data_sdv)

    print(f"   Trainiere CTGAN ({epochs} Epochen) …")
    sdv_model = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    sdv_model.fit(train_data_sdv)

    synthetic_sdv = sdv_model.sample(num_rows=len(train_data_sdv))
    y_synth       = le.transform(synthetic_sdv[target_col].str.strip())
    X_synth_raw   = synthetic_sdv.drop(target_col, axis=1)
    X_synth       = pd.get_dummies(X_synth_raw).reindex(columns=X_train_cols, fill_value=0)

    return X_synth, y_synth
