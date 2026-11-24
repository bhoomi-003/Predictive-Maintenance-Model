import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# PURPOSE
# The original AI4I 2020 dataset has 0 missing values and 0 duplicate
# rows, so our cleaning steps had nothing to do. To demonstrate a
# complete cleaning pipeline, this script makes a MODIFIED copy of the
# dataset with a small, controlled amount of:
#   1. missing values (NaN) in the feature columns
#   2. exact duplicate rows
#
# The original file (data/ai4i2020.csv) is NOT changed.
# Output: data/ai4i2020_modified.csv
#
# NOTE: Class imbalance is NOT fixed in this file on purpose.
# Balancing must be done only on the TRAINING set, after the
# train/validation/test split (see datacleaning.py for the reason).
# ------------------------------------------------------------------

# Fixed seed so every team member gets exactly the same modified file.
rng = np.random.default_rng(42)

df = pd.read_csv("data/ai4i2020.csv")
modified_df = df.copy()


# -------------------- 1. Inject Missing Values --------------------

# Only feature columns get missing values.
# We never blank out the target ("Machine failure"), identifiers, or
# the failure-type flags (those are dropped later anyway).
# ~1% of rows in each of these columns become NaN (100 cells each).

columns_with_missing = {
    "Type": 100,
    "Air temperature [K]": 100,
    "Process temperature [K]": 100,
    "Rotational speed [rpm]": 100,
    "Torque [Nm]": 100,
    "Tool wear [min]": 100,
}

for column, n_missing in columns_with_missing.items():
    # Pick random row positions independently for each column.
    rows = rng.choice(len(modified_df), size=n_missing, replace=False)
    modified_df.loc[rows, column] = np.nan


# -------------------- 2. Inject Duplicate Rows --------------------

# Copy 150 random rows (1.5% of the data) and append them again.
# They are exact copies (same UDI, same values, including any NaNs),
# which is what a double-logged sensor record looks like.

duplicate_rows = rng.choice(len(modified_df), size=150, replace=False)
duplicates = modified_df.iloc[duplicate_rows]

modified_df = pd.concat([modified_df, duplicates], ignore_index=True)

# Sort by UDI so each duplicate sits right after its original row.
modified_df = modified_df.sort_values("UDI", kind="stable").reset_index(drop=True)


# -------------------- Save and Report --------------------

modified_df.to_csv("data/ai4i2020_modified.csv", index=False)

print("Original shape:", df.shape)
print("Modified shape:", modified_df.shape)

print("\nMissing values per column (modified):")
print(modified_df.isnull().sum())

print("\nDuplicate rows (modified):", modified_df.duplicated().sum())

print("\nMachine failure distribution (modified):")
print(modified_df["Machine failure"].value_counts())
