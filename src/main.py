import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

import os

# Daten laden
df = pd.read_csv(
    "data//adult/adult.data",
    header=None,
    sep=",",
    skipinitialspace=True
)
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

# df preparen
df = df.replace("?", pd.NA)
df = df.dropna()
df["income"] = df["income"].str.strip()
y = df["income"]
X = df.drop("income", axis=1)

# kategorische Features encoden
X = pd.get_dummies(X)

le = LabelEncoder()
y = le.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== BASELINE (REAL DATA) ============================")
print(classification_report(y_test, y_pred, zero_division=0))

# SMOTE anwenden
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("======= CLASS DISTRIBUTION =======")
print("Train (before SMOTE):")
print(pd.Series(y_train).value_counts())
print("\nTrain (after SMOTE):")
print(pd.Series(y_resampled).value_counts())

# Modell jz mit SMOTE trainieren
model_smote = RandomForestClassifier()
model_smote.fit(X_resampled, y_resampled)

y_pred_smote = model_smote.predict(X_test)

print("=== SMOTE MODEL =====================================")
print(classification_report(y_test, y_pred_smote, zero_division=0))

report_base = classification_report(y_test, y_pred, output_dict=True)
report_smote = classification_report(y_test, y_pred_smote, output_dict=True)


#SDV
# Train-Daten
train_data = X_train.copy()
train_data["income"] = y_train

# Metadata erkennen
metadata = Metadata.detect_from_dataframe(data=train_data)
metadata.detect_from_dataframe(train_data)

# Datenstruktur speichern
os.remove("artifacts/metadata.json") # später removen
metadata.save_to_json("artifacts/metadata.json")

# Modell erstellen
sdv_model = CTGANSynthesizer(metadata, epochs=10) # === === === === === === === ===========================================

# Train
sdv_model.fit(train_data)

# SD erzeugen
synthetic_data = sdv_model.sample(num_rows=len(train_data))

#print(synthetic_data.head())

# trennen
X_synth = synthetic_data.drop("income", axis=1)
y_synth = synthetic_data["income"]

# train
model_sdv = RandomForestClassifier()
model_sdv.fit(X_synth, y_synth)

# test
y_pred_sdv = model_sdv.predict(X_test)


print("=== SDV MODEL =======================================")
print(classification_report(y_test, y_pred_sdv, zero_division=0))


report_base = classification_report(y_test, y_pred, output_dict=True)
report_smote = classification_report(y_test, y_pred_smote, output_dict=True)
report_sdv = classification_report(y_test, y_pred_sdv, output_dict=True)

print("======= MINORITY CLASS FOCUS =======")
print("Baseline Recall (class 1):", report_base["1"]["recall"])
print("SMOTE Recall (class 1)   :", report_smote["1"]["recall"])
print("SDV Recall (class 1)     :", report_sdv["1"]["recall"])

print("\nBaseline F1 (class 1):", report_base["1"]["f1-score"])
print("SMOTE F1 (class 1)   :", report_smote["1"]["f1-score"])
print("SDV F1 (class 1)     :", report_sdv["1"]["f1-score"])