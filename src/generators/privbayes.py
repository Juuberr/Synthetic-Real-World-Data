import os
import tempfile

import numpy as np
import pandas as pd

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


DEFAULT_NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

DEFAULT_CORE_COLS = [
    "age",
    "education_num",
    "hours_per_week",
    "education",
    "marital_status",
    "occupation",
    "sex",
    "income",
]


def _sample_marginal(series, n_rows, rng):
    values = series.dropna().to_numpy()
    if len(values) == 0:
        return [pd.NA] * n_rows
    return rng.choice(values, size=n_rows, replace=True)


def generate(
    X_raw,
    y_raw,
    train_index,
    X_train_cols,
    seed,
    **kwargs,
):
    """PrivBayes via DataSynthesizer correlated_attribute_mode.

    DataSynthesizer's full Bayesian-network search is slow on the complete
    Adult schema, so PrivBayes learns dependencies on a high-signal core
    schema and fills the remaining feature columns from training marginals.
    """
    le = kwargs["le"]
    target_col = kwargs.get("target_col", "income")
    epsilon = kwargs.get("epsilon", 1.0)
    k = kwargs.get("k", 1)
    numeric_cols = kwargs.get("numeric_cols", DEFAULT_NUMERIC_COLS)
    core_cols = kwargs.get("core_cols", DEFAULT_CORE_COLS)

    train_features = X_raw.loc[train_index].copy()
    output_rows = len(train_features)
    train_full = train_features.copy()
    train_full[target_col] = y_raw.loc[train_index].values

    core_cols = [col for col in core_cols if col == target_col or col in train_full.columns]
    if target_col not in core_cols:
        core_cols.append(target_col)
    core_df = train_full[core_cols].copy()

    attribute_to_is_categorical = {
        col: col not in numeric_cols
        for col in core_df.columns
    }

    print(
        f"   Trainiere PrivBayes (epsilon={epsilon}, k={k}, "
        f"{len(core_cols)} Kernspalten) ..."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        real_csv = os.path.join(tmpdir, "real.csv")
        description_json = os.path.join(tmpdir, "description.json")
        synthetic_csv = os.path.join(tmpdir, "synthetic.csv")

        core_df.to_csv(real_csv, index=False)

        describer = DataDescriber(category_threshold=100)
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=real_csv,
            epsilon=epsilon,
            k=k,
            attribute_to_is_categorical=attribute_to_is_categorical,
            seed=seed,
        )
        describer.save_dataset_description_to_file(description_json)

        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(
            output_rows,
            description_json,
            seed=seed,
        )
        generator.save_synthetic_data(synthetic_csv)

        synthetic_core = pd.read_csv(synthetic_csv)

    rng = np.random.default_rng(seed)
    synthetic_features = pd.DataFrame(index=range(output_rows))
    for col in X_raw.columns:
        if col in synthetic_core.columns:
            synthetic_features[col] = synthetic_core[col].values
        else:
            synthetic_features[col] = _sample_marginal(train_features[col], output_rows, rng)

    y_synth_raw = synthetic_core[target_col].astype(str).str.strip()
    known_labels = set(le.classes_)
    invalid = ~y_synth_raw.isin(known_labels)
    if invalid.any():
        fallback = rng.choice(le.classes_, size=int(invalid.sum()), replace=True)
        y_synth_raw.loc[invalid] = fallback

    y_synth = le.transform(y_synth_raw)
    X_synth = (
        pd.get_dummies(synthetic_features)
        .reindex(columns=X_train_cols, fill_value=0)
    )

    return X_synth, y_synth
