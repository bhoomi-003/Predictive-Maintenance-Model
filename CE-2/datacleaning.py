# ==================================================================
# CHANGES FROM THE CE1 VERSION OF THIS FILE
#
# [CHANGE 1] Input file: now loads data/ai4i2020_modified.csv
#            (made by create_modified_dataset.py) instead of the
#            original ai4i2020.csv, which had no missing values or
#            duplicates to clean.
# [CHANGE 2] Missing-value and duplicate checks are now active code
#            (they were commented out), and duplicates are removed.
# [CHANGE 3] Train/test split replaced by a stratified
#            train / validation / test split (70 / 15 / 15), done
#            BEFORE imputation and balancing (CE1 feedback).
# [CHANGE 4] NEW: missing values filled with median (numeric) and
#            mode (Type), learned from the training set only.
# [CHANGE 5] NEW: class imbalance handled with SMOTENC, applied to
#            the training set only (CE1 feedback).
# [CHANGE 6] One-hot encoding moved AFTER the split and done the same
#            way for all three sets (fixed categories H, L, M).
# [CHANGE 7] NEW: the final train / validation / test sets are saved
#            as CSV files for the model-training step.
# [CHANGE 8] NEW: also saves the training set BEFORE balancing
#            (data/train_unbalanced.csv), needed for cross-validation
#            in modeltuning.py.
# [CHANGE 9] NEW: feature engineering. 3 new features (temperature
#            difference, power, torque x tool wear) are added after
#            SMOTENC, giving 11 input features instead of 8.
#
# Unchanged from CE1: feature selection (UDI, Product ID and the
# TWF/HDF/PWF/OSF/RNF leakage columns are still dropped).
# ==================================================================

import pandas as pd

# Splits data into training / validation / testing sets.
from sklearn.model_selection import train_test_split

# [CHANGE 4] NEW import.
# Fills missing values using statistics learned from the training set.
from sklearn.impute import SimpleImputer

# [CHANGE 5] NEW import.
# SMOTENC = SMOTE for datasets that contain a categorical column ("Type").
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTENC

# [CHANGE 9] NEW import. Our own file (features.py, same folder) that
# creates the 3 engineered features.
from features import add_features, ENGINEERED_COLUMNS


# ==================================================================
# CORRECT ORDER OF STEPS (this is the main fix from CE1 feedback)
#
#   1. Load data
#   2. Remove duplicate rows                 -> on FULL data (before split)
#   3. Drop ID + leakage columns, make X, y
#   4. Split into train / validation / test  -> stratified
#   5. Fill missing values                   -> learned from TRAIN only
#   6. Balance classes (SMOTENC)             -> TRAIN only
#   6b. Add engineered features              -> all sets, same formulas
#   7. One-hot encode "Type"
#   8. Save the three sets
#
# Rule: anything that LEARNS from the data (median, mode, SMOTE
# neighbours, later the scaler and the model) must only see the
# training set. Validation and test must stay like real, unseen data.
# ==================================================================


# -------------------- 1. Load Data --------------------

# [CHANGE 1] CE1 loaded "data/ai4i2020.csv". We now load the modified
# copy that contains missing values and duplicate rows.
df = pd.read_csv("data/ai4i2020_modified.csv")

print("Dataset Shape:", df.shape)


# -------------------- 2. Missing Values and Duplicates Check --------------------

# [CHANGE 2] In CE1 these checks were commented out because the
# result was 0. They now run and report the problems that exist in
# the modified dataset. Missing values are only COUNTED here; they are
# FILLED in step 5, after the split.

print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())
print("Rows with at least one Missing Value:", df.isnull().any(axis=1).sum())

print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# Remove duplicates BEFORE splitting.
# If we split first, one copy of a row could land in train and the
# other copy in test. The model would then be tested on a row it has
# already seen, and the test score would look better than it really is.
df = df.drop_duplicates().reset_index(drop=True)

print("Shape after removing Duplicates:", df.shape)

# The target must never be missing (we cannot "guess" a label).
# If any row had no label, we would drop it here.
df = df.dropna(subset=["Machine failure"])


# -------------------- 3. Feature Selection --------------------

# Unchanged from CE1: same columns kept and dropped.
# Small change: X and y are now made from the RAW "Type" column.
# One-hot encoding happens later (step 7) instead of here.

# UDI and Product ID are removed because they are identification columns.
# TWF, HDF, PWF, OSF, and RNF are removed to avoid data leakage
# (they are sub-flags of the target itself - see CE1 analysis).

feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
numeric_columns = feature_columns[1:]

X = df[feature_columns].copy()
y = df["Machine failure"].astype(int)


# -------------------- 4. Train / Validation / Test Split --------------------

# [CHANGE 3] CE1 did a single 80/20 train/test split as the LAST step,
# after everything else was done to the full dataset. Two problems:
#   - No validation set: if we compare models or tune them on the
#     test set, the final test score is no longer an honest estimate.
#   - Wrong position: steps that learn from data (imputation,
#     balancing, scaling) must come AFTER the split and be fitted on
#     train only. So the split now happens right after cleaning.

# 70% train, 15% validation, 15% test.
#
# Train      -> the model learns from this (and only this gets balanced).
# Validation -> used to compare models and tune settings.
# Test       -> used ONCE at the very end for the final reported score.
#
# stratify keeps the original ~3.4% failure rate in every set, so
# validation and test reflect the real-world class imbalance.

# Step A: separate the 15% test set.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# Step B: split the remaining 85% into train (70%) and validation (15%).
# 0.15 / 0.85 = 0.1765 of the remaining data.
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=y_temp
)

print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)


# -------------------- 5. Handle Missing Values --------------------

# [CHANGE 4] NEW step. CE1 needed no imputation (0 missing values).

# Numeric columns -> median (robust to outliers, e.g. torque/tool wear).
# "Type"          -> most frequent value (mode).
#
# fit() is called on TRAIN only. The same train median/mode is then
# used to fill validation and test. Computing the median on the full
# dataset would leak information from the test set into training.

numeric_imputer = SimpleImputer(strategy="median")
type_imputer = SimpleImputer(strategy="most_frequent")

numeric_imputer.fit(X_train[numeric_columns])
type_imputer.fit(X_train[["Type"]])

print("\nMedians learned from Train:")
print(pd.Series(numeric_imputer.statistics_, index=numeric_columns))
print("Mode of Type learned from Train:", type_imputer.statistics_[0])


def fill_missing(data):
    data = data.copy()
    data[numeric_columns] = numeric_imputer.transform(data[numeric_columns])
    data[["Type"]] = type_imputer.transform(data[["Type"]])
    return data


X_train = fill_missing(X_train)
X_val = fill_missing(X_val)
X_test = fill_missing(X_test)

print("\nMissing Values after Imputation (train / val / test):",
      X_train.isnull().sum().sum(),
      X_val.isnull().sum().sum(),
      X_test.isnull().sum().sum())


# -------------------- 6. Handle Class Imbalance (Train Only) --------------------

# [CHANGE 5] NEW step (CE1 feedback). In CE1 the training set had
# only ~3.4% failures, so a model could score ~96.6% accuracy by
# always predicting "no failure". We now balance the training set.

# SMOTE creates NEW synthetic failure rows by interpolating between a
# real failure row and its nearest failure neighbours.
# SMOTENC is used because "Type" is categorical: it picks a real
# category (L/M/H) instead of making values like Type = 0.4.
#
# Applied ONLY to the training set:
# - Synthetic rows must never appear in validation/test, otherwise
#   we would be scoring the model on fake data.
# - Validation/test must keep the real ~3.4% failure rate so the
#   scores tell us how the model behaves in a real factory.

print("\nTrain Failure Distribution BEFORE SMOTE:")
print(y_train.value_counts())

smote = SMOTENC(
    categorical_features=["Type"],
    sampling_strategy=1.0,   # 1.0 -> equal number of failures and non-failures
    random_state=42
)

X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# [CHANGE 8] Keep a copy of the training set BEFORE balancing.
# modeltuning.py needs it for cross-validation: SMOTENC must be applied
# inside each fold, so the folds must start from real (unbalanced) rows.
X_train_unbal = X_train.copy()
y_train_unbal = y_train.copy()

# Rotational speed and tool wear are whole numbers in the original
# data, so round the synthetic values to keep them realistic.
X_train_bal["Rotational speed [rpm]"] = X_train_bal["Rotational speed [rpm]"].round()
X_train_bal["Tool wear [min]"] = X_train_bal["Tool wear [min]"].round()
# Same for the imputed medians in all sets.
for data in (X_val, X_test):
    data["Rotational speed [rpm]"] = data["Rotational speed [rpm]"].round()
    data["Tool wear [min]"] = data["Tool wear [min]"].round()

print("\nTrain Failure Distribution AFTER SMOTE:")
print(y_train_bal.value_counts())


# -------------------- 6b. Feature Engineering --------------------

# [CHANGE 9] NEW step. Adds 3 features calculated from the sensor
# readings (formulas and reasons are in features.py):
#   Temp difference [K]  = Process temperature - Air temperature
#   Power [W]            = Torque x Rotational speed x 2 x pi / 60
#   Torque x Tool wear   = Torque x Tool wear
#
# Why AFTER SMOTENC: SMOTENC creates synthetic rows by mixing values of
# two real rows. If the new features were added before it, a synthetic
# row's "Power" would be a mix of two powers instead of its own torque
# x speed. Adding them afterwards keeps every row physically correct.
#
# Why this is safe (no leakage): each value comes only from the other
# columns of the same row. Nothing is learned from the data, so it can
# be applied to every set in the same way.
#
# Effect (Random Forest, validation set, default settings): F1 rose from
# about 0.50 to 0.62. See modeltraining.py / modeltuning.py for results.

X_train_bal = add_features(X_train_bal)
X_train_unbal = add_features(X_train_unbal)
X_val = add_features(X_val)
X_test = add_features(X_test)

print("\nEngineered features (first 5 training rows):")
print(X_train_bal[ENGINEERED_COLUMNS].head())


# -------------------- 7. One-Hot Encoding --------------------

# [CHANGE 6] Same idea as CE1 (Type -> Type_H, Type_L, Type_M), but
# moved here so it runs after imputation and SMOTENC. (SMOTENC needs
# "Type" as one categorical column, not three 0/1 columns.)

# Fixing the categories guarantees all three sets get the same
# columns (Type_H, Type_L, Type_M) in the same order.

def encode(data):
    data = data.copy()
    data["Type"] = pd.Categorical(data["Type"], categories=["H", "L", "M"])
    return pd.get_dummies(data, columns=["Type"], dtype=int)


X_train_bal = encode(X_train_bal)
X_train_unbal = encode(X_train_unbal)   # [CHANGE 8]
X_val = encode(X_val)
X_test = encode(X_test)

print("\nColumns After Encoding:")
print(list(X_train_bal.columns))


# -------------------- 8. Save --------------------

# [CHANGE 7] NEW step. Saves the prepared sets so the model-training
# script can load them directly.

train_out = X_train_bal.assign(**{"Machine failure": y_train_bal.values})
val_out = X_val.assign(**{"Machine failure": y_val.values})
test_out = X_test.assign(**{"Machine failure": y_test.values})

# Shows the first 5 rows of the final balanced training set (for the PPT screenshot).
print("\nFinal Balanced Training Set (first 5 rows):")
print(train_out.head())

train_out.to_csv("data/train_balanced.csv", index=False)

# [CHANGE 8] NEW: training set after imputation but BEFORE SMOTENC
# (7000 rows, real failure rate). Used only for cross-validation in
# modeltuning.py.
X_train_unbal.assign(**{"Machine failure": y_train_unbal.values}).to_csv(
    "data/train_unbalanced.csv", index=False
)
val_out.to_csv("data/validation.csv", index=False)
test_out.to_csv("data/test.csv", index=False)

print("\nFinal Shapes:")
print("Train (balanced):", train_out.shape)
print("Validation      :", val_out.shape)
print("Test            :", test_out.shape)

print("\nFailure Distribution - Validation:")
print(y_val.value_counts())
print("\nFailure Distribution - Test:")
print(y_test.value_counts())

# Next step (model training): if a model needs scaling (e.g. Logistic
# Regression, SVM, KNN), fit the StandardScaler on train_balanced only,
# then transform validation and test with it.
