import os
import tempfile

import pandas as pd

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator



def generate(
    X_raw,
    y_raw,
    train_index,
    X_train_cols,
    seed,
    **kwargs,
):
    """
    PrivBayes Generator (DataSynthesizer).

    Input:
        rohe Trainingsdaten

    Output:
        X_synth
        y_synth
    """

    print(">>> PrivBayes wird ausgeführt")

    le = kwargs["le"]
    target_col = kwargs.get("target_col", "income")

    train_df = X_raw.loc[train_index].copy()
    train_df[target_col] = y_raw.loc[train_index].values

    print("   Trainiere PrivBayes ...")

    with tempfile.TemporaryDirectory() as tmpdir:

        real_csv = os.path.join(tmpdir, "real.csv")
        description_json = os.path.join(tmpdir, "description.json")
        synthetic_csv = os.path.join(tmpdir, "synthetic.csv")

        train_df.to_csv(real_csv, index=False)

        describer = DataDescriber(category_threshold=20)

        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=real_csv,
            epsilon=1.0,
            k=2,
        )

        describer.save_dataset_description_to_file(
            description_json
        )

        generator = DataGenerator()

        generator.generate_dataset_in_correlated_attribute_mode(
            len(train_df),
            description_json,
        )

        generator.save_synthetic_data(
            synthetic_csv
        )

        synthetic = pd.read_csv(
            synthetic_csv
        )

    y_synth = le.transform(
        synthetic[target_col]
        .astype(str)
        .str.strip()
    )

    X_synth_raw = synthetic.drop(
        columns=[target_col]
    )

    X_synth = (
        pd.get_dummies(X_synth_raw)
        .reindex(columns=X_train_cols, fill_value=0)
    )

    return X_synth, y_synth