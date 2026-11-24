# ==================================================================
# RANDOM FOREST: TUNING, THRESHOLD, FINAL TEST (NEW FILE)
#
# Random Forest was the best model in modeltraining.py (highest F1 on
# the validation set) but it overfitted (training recall = 1.0).
# Uses 11 features: 8 original + 3 engineered (features.py).
# This file:
#   1. Tunes its settings with 5-fold cross-validation
#   2. Picks the best decision threshold on the validation set
#   3. Evaluates the final model ONCE on the test set
#   4. Shows which features matter most (feature importance)
#   5. Saves the final model
#
# Input files (made by datacleaning.py):
#   data/train_unbalanced.csv -> cross-validation + final training
#   data/validation.csv       -> threshold selection, overfitting check
#   data/test.csv             -> final score, used ONCE at the end
#
# Run order: ... -> datacleaning.py -> modeltraining.py -> modeltuning.py
# ==================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # saves the trained model to a file (installed with scikit-learn)

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# imblearn's Pipeline (not sklearn's) is needed because it allows a
# resampling step (SMOTENC) inside the pipeline.
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

# Runs our own add_features() function as a step inside the pipeline.
from sklearn.preprocessing import FunctionTransformer

# Our own file with the 3 engineered features (same formulas as datacleaning.py).
from features import add_features, ENGINEERED_COLUMNS

pd.set_option("display.width", 200)


# -------------------- 1. Load Data --------------------

train_df = pd.read_csv("data/train_unbalanced.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

type_columns = ["Type_H", "Type_L", "Type_M"]


def prepare(df):
    """
    Turns the three one-hot columns back into one "Type" column (H/L/M).

    Why: SMOTENC needs "Type" as ONE categorical column so it can give
    each synthetic row a real machine type. With three 0/1 columns it
    could create impossible rows (e.g. Type_H = 1 AND Type_L = 1).
    One-hot encoding is done again inside the pipeline, after SMOTENC.
    """
    X = df.drop(columns=["Machine failure"]).copy()
    X["Type"] = X[type_columns].idxmax(axis=1).str.replace("Type_", "")
    X = X.drop(columns=type_columns)

    # The 3 engineered features are removed here and calculated again
    # inside the pipeline, AFTER SMOTENC (same reason as in
    # datacleaning.py: synthetic rows must get their own correct
    # power / temperature difference, not a mix of two rows' values).
    X = X.drop(columns=ENGINEERED_COLUMNS)
    y = df["Machine failure"]
    return X, y


X_train, y_train = prepare(train_df)
X_val, y_val = prepare(val_df)
X_test, y_test = prepare(test_df)

print("Train (unbalanced):", X_train.shape, "| failures:", y_train.sum())
print("Validation        :", X_val.shape, "| failures:", y_val.sum())
print("Test              :", X_test.shape, "| failures:", y_test.sum())


# -------------------- 2. Build the Pipeline --------------------

# The pipeline runs these steps in order every time it is trained:
#   1. SMOTENC  -> balances the data it is given (1:1, as in datacleaning.py)
#   2. Features -> adds the 3 engineered features (features.py)
#   3. Encode   -> one-hot encodes Type into Type_H, Type_L, Type_M
#   4. Model    -> Random Forest
#
# Why a pipeline: during cross-validation, each fold is split into a
# training part and a scoring part. SMOTENC must only see the
# training part. If we used train_balanced.csv instead, synthetic rows
# made from a real failure could end up in the scoring part while
# their "parent" row is in the training part, and the scores would
# look better than they really are. Inside the pipeline, SMOTENC is
# re-run on the training part of every fold only.
#
# During prediction (validation / test), SMOTENC is skipped
# automatically: resampling only happens while training. The feature
# and encoding steps still run, so new data only needs the original
# 5 sensor readings + Type.

feature_step = FunctionTransformer(add_features)

encoder = ColumnTransformer(
    [("type", OneHotEncoder(categories=[["H", "L", "M"]]), ["Type"])],
    remainder="passthrough"  # keep the sensor + engineered columns as they are
)

pipeline = Pipeline([
    ("smote", SMOTENC(categorical_features=["Type"], sampling_strategy=1.0, random_state=42)),
    ("features", feature_step),
    ("encode", encoder),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
])


# -------------------- 3. Hyper-parameter Tuning (Grid Search + Cross-Validation) --------------------

# Settings we try (every combination = 2 x 4 x 3 = 24 combinations):
#
# n_estimators     -> number of trees. More trees = more stable, slower.
# max_depth        -> how deep each tree can grow. None = no limit,
#                     which lets trees memorise the training data
#                     (the overfitting we saw). Smaller = simpler trees.
# min_samples_leaf -> minimum rows allowed in a final leaf. Larger
#                     values stop trees from making rules for just 1 row.
#
# ("rf__" in front means "this setting belongs to the rf step".)

param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 10, 15, 20],
    "rf__min_samples_leaf": [1, 3, 5],
}

# 5-fold stratified cross-validation:
# the training set is split into 5 parts with the same failure rate.
# Each combination is trained 5 times, each time scored on a different
# part, and the 5 F1-scores are averaged. 24 x 5 = 120 training runs.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1",          # F1 of the failure class
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

print("\nRunning grid search (24 combinations x 5 folds), this can take a few minutes...")
grid.fit(X_train, y_train)

# Top 5 combinations by average cross-validation F1.
cv_results = pd.DataFrame(grid.cv_results_)
top5 = cv_results.sort_values("mean_test_score", ascending=False).head(5)
print("\nTop 5 settings (mean F1 across 5 folds):")
print(top5[["param_rf__n_estimators", "param_rf__max_depth", "param_rf__min_samples_leaf",
            "mean_train_score", "mean_test_score", "std_test_score"]].round(3).to_string(index=False))

# RESULT: the default settings (no depth limit, 100 trees, leaf size 1)
# scored best. Limiting tree depth lowered the cross-validation F1, so
# the perfect training score is normal for a Random Forest here and is
# not what limits its performance. The decision threshold (step 6) and
# the engineered features made the real difference.

print("\nBest settings:", grid.best_params_)
print("Best cross-validation F1:", round(grid.best_score_, 3))

# GridSearchCV automatically retrains the best settings on the WHOLE
# training set (SMOTENC applied once more inside the pipeline).
best_model = grid.best_estimator_


# -------------------- 4. Tuned vs Untuned on Validation --------------------

# Compare against the default Random Forest (the one from modeltraining.py),
# built the same way so the comparison is fair.
default_model = Pipeline([
    ("smote", SMOTENC(categorical_features=["Type"], sampling_strategy=1.0, random_state=42)),
    ("features", feature_step),
    ("encode", encoder),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
default_model.fit(X_train, y_train)


def scores(model, X, y, threshold=0.5):
    """Precision, recall, F1 for the failure class at a given threshold."""
    proba = model.predict_proba(X)[:, 1]   # probability of failure
    pred = (proba >= threshold).astype(int)
    return {
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1-score": f1_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
    }


comparison = pd.DataFrame({
    "Default RF": scores(default_model, X_val, y_val),
    "Tuned RF": scores(best_model, X_val, y_val),
}).T
print("\nValidation Set, threshold 0.5:")
print(comparison.round(3))


# -------------------- 5. Overfitting Check --------------------

# Recall on the (real, unbalanced) training rows vs recall on validation.
# Recall is used because it does not depend on the class balance
# (see modeltraining.py, section 6).
for name, model in [("Default RF", default_model), ("Tuned RF", best_model)]:
    tr = recall_score(y_train, model.predict(X_train))
    va = recall_score(y_val, model.predict(X_val))
    print(f"{name}: train recall = {tr:.3f}, validation recall = {va:.3f}, gap = {tr - va:.3f}")


# -------------------- 6. Decision Threshold --------------------

# By default a model says "failure" when the failure probability is
# >= 0.5. That cut-off is not always the best one. A lower threshold
# catches more failures (higher recall) but raises more false alarms
# (lower precision); a higher threshold does the opposite.
#
# We try thresholds from 0.10 to 0.90 on the VALIDATION set and keep
# the one with the highest F1. The test set is NOT used to choose it,
# otherwise the final test score would no longer be honest.

val_proba = best_model.predict_proba(X_val)[:, 1]
thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

threshold_rows = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    threshold_rows.append({
        "Threshold": t,
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred),
        "F1-score": f1_score(y_val, pred),
    })
threshold_df = pd.DataFrame(threshold_rows)

best_threshold = threshold_df.loc[threshold_df["F1-score"].idxmax(), "Threshold"]

print("\nThreshold search on validation set:")
print(threshold_df.round(3).to_string(index=False))
print("\nChosen threshold:", best_threshold)

# Plot: how precision, recall and F1 change with the threshold.
plt.figure(figsize=(8, 5))
plt.plot(threshold_df["Threshold"], threshold_df["Precision"], marker="o", label="Precision", color="#4E6A9A")
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], marker="o", label="Recall", color="#D2803A")
plt.plot(threshold_df["Threshold"], threshold_df["F1-score"], marker="o", label="F1-score", color="#A9443A")
plt.axvline(best_threshold, linestyle="--", color="grey", label=f"Chosen = {best_threshold}")
plt.title("Precision / Recall / F1 vs Decision Threshold (Validation Set)")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------- 7. Final Evaluation on the Test Set (ONCE) --------------------

# This is the first and only time the test set is used.
# Settings and threshold were all chosen without it, so this score is
# an honest estimate of how the system would do on new machines.

test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print("\n" + "=" * 60)
print(f"FINAL TEST RESULT (Tuned Random Forest, threshold = {best_threshold})")
print("=" * 60)
print("Confusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, test_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test, test_pred, target_names=["No Failure", "Failure"], digits=3))

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, test_pred),
    display_labels=["No Failure", "Failure"]
).plot(cmap="Blues", colorbar=False)
plt.title(f"Final Model on Test Set (threshold = {best_threshold})")
plt.tight_layout()
plt.show()


# -------------------- 8. Feature Importance --------------------

# How much each feature helped the Random Forest separate failures
# from healthy machines (all values add up to 1).
# The three Type columns are added together into one "Type" value.

rf = best_model.named_steps["rf"]
feature_names = best_model.named_steps["encode"].get_feature_names_out()
importance = pd.Series(rf.feature_importances_, index=feature_names)

# Clean names: "type__Type_H" -> "Type_H", "remainder__Torque [Nm]" -> "Torque [Nm]"
importance.index = [n.split("__", 1)[1] for n in importance.index]
type_total = importance[["Type_H", "Type_L", "Type_M"]].sum()
importance = importance.drop(["Type_H", "Type_L", "Type_M"])
importance["Type (H/L/M)"] = type_total
importance = importance.sort_values()

print("\nFeature Importance:")
print(importance.sort_values(ascending=False).round(3))

plt.figure(figsize=(8, 5))
plt.barh(importance.index, importance.values, color="#4E6A9A")
for i, v in enumerate(importance.values):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.xlim(0, importance.max() * 1.15)  # leave room for the value labels
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# -------------------- 9. Save the Final Model --------------------

# Saves the whole pipeline (feature engineering + encoding + Random
# Forest) and the chosen
# threshold, so the system can predict on new sensor readings later
# without retraining.
os.makedirs("models", exist_ok=True)
joblib.dump({"model": best_model, "threshold": best_threshold}, "models/random_forest_final.pkl")
print("\nSaved final model to models/random_forest_final.pkl")
