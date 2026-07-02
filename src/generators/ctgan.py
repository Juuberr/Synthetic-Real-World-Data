import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer


def generate(X_raw, y_raw, train_index, X_train_cols, seed, **kwargs):
    """CTGAN synthesizer using raw categorical columns.

    The model trains on raw Adult columns and one-hot encodes only after
    sampling, so SDV can model categoricals instead of binary dummy columns.
    """
    le = kwargs["le"]
    target_col = kwargs.get("target_col", "income")

    epochs = kwargs.get("epochs", 20)
    max_train_rows = kwargs.get("max_train_rows", 3000)
    batch_size = kwargs.get("batch_size", 500)
    generator_dim = kwargs.get("generator_dim", (256, 256))
    discriminator_dim = kwargs.get("discriminator_dim", (256, 256))
    discriminator_steps = kwargs.get("discriminator_steps", 1)
    pac = kwargs.get("pac", 10)

    train_data = X_raw.loc[train_index].copy()
    train_data[target_col] = y_raw.loc[train_index].values
    output_rows = len(train_data)

    if max_train_rows and len(train_data) > max_train_rows:
        train_data = train_data.sample(n=max_train_rows, random_state=seed)
        print(f"   CTGAN-Training auf {len(train_data):,} von {output_rows:,} Trainingszeilen")

    metadata = Metadata.detect_from_dataframe(data=train_data)

    print(
        f"   Trainiere CTGAN: epochs={epochs}, batch_size={batch_size}, "
        f"generator_dim={generator_dim}, discriminator_dim={discriminator_dim}, "
        f"discriminator_steps={discriminator_steps}"
    )
    model = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        discriminator_steps=discriminator_steps,
        pac=pac,
        verbose=False,
    )
    model.fit(train_data)

    synthetic = model.sample(num_rows=output_rows)
    y_synth = le.transform(synthetic[target_col].str.strip())
    X_synth_raw = synthetic.drop(target_col, axis=1)
    X_synth = pd.get_dummies(X_synth_raw).reindex(columns=X_train_cols, fill_value=0)

    return X_synth, y_synth
