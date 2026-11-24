# ==================================================================
# MODEL TRAINING AND COMPARISON (NEW FILE)
#
# Trains the six classification models from our syllabus
# (Lectures 16-18) and compares them on the VALIDATION set:
#   1. Logistic Regression
#   2. K-Nearest Neighbour (KNN)
#   3. Decision Tree
#   4. Random Forest
#   5. Naive Bayes (Gaussian)
#   6. Support Vector Machine (SVM)
#
# Input files (made by datacleaning.py):
#   data/train_balanced.csv -> used to TRAIN the models
#   data/validation.csv     -> used to COMPARE the models
#   data/test.csv           -> NOT used here. It is kept untouched and
#                              used only once at the very end, on the
#                              final chosen model.
#
# Run order: create_modified_dataset.py -> datapreprocessing.py ->
#            plottingdata.py -> datacleaning.py -> modeltraining.py
# ==================================================================

import pandas as pd
import matplotlib.pyplot as plt

# Feature scaling (standardization).
from sklearn.preprocessing import StandardScaler

# The six classification models.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Evaluation metrics (Lectures 19-21).
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

pd.set_option("display.width", 200)  # print wide tables on one line


# -------------------- 1. Load Data --------------------

train_df = pd.read_csv("data/train_balanced.csv")
val_df = pd.read_csv("data/validation.csv")

# Separate input features (X) from the target (y).
X_train = train_df.drop(columns=["Machine failure"])
y_train = train_df["Machine failure"]

X_val = val_df.drop(columns=["Machine failure"])
y_val = val_df["Machine failure"]

print("Train Shape:", X_train.shape)        # (13526, 8) - balanced with SMOTENC
print("Validation Shape:", X_val.shape)     # (1500, 8)  - real failure rate (~3.4%)

print("\nTrain Failure Distribution:")
print(y_train.value_counts())
print("\nValidation Failure Distribution:")
print(y_val.value_counts())


# -------------------- 2. Feature Scaling --------------------

# Standardization: changes each feature to mean 0 and standard deviation 1.
#
# Why: features have very different ranges, e.g. Rotational speed is
# ~1500 rpm but Torque is ~40 Nm. Models that use distances or weights
# (Logistic Regression, KNN, SVM) would be dominated by the large
# numbers without scaling.
#
# Tree models (Decision Tree, Random Forest) and Naive Bayes do not
# need scaling, so they use the original (unscaled) values.
#
# fit() is done on the TRAINING set only. The same mean and standard
# deviation are then used to transform the validation set. Fitting on
# validation data would leak information about it into the model.

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)


# -------------------- 3. Define the Models --------------------

# random_state=42 makes results the same every time the code runs.
# All models use default / simple settings for now. Tuning comes later,
# only for the best model.
#
# Each entry: (model, uses_scaled_data)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "KNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
    "Naive Bayes": (GaussianNB(), False),
    "SVM": (SVC(kernel="rbf", random_state=42), True),
}


# -------------------- 4. Train and Evaluate --------------------

# Metrics are calculated for the FAILURE class (Machine failure = 1),
# because that is the class we care about:
#
# Precision -> of the machines the model flagged as "will fail",
#              how many really failed? (low precision = many false alarms)
# Recall    -> of the machines that really failed, how many did the
#              model catch? (low recall = missed failures, the costly mistake)
# F1-score  -> balance of precision and recall (main metric for ranking).
# Accuracy  -> shown for reference only. On validation, a model that
#              always says "no failure" would already get ~96.6%.

results = []
confusion_matrices = {}

for name, (model, uses_scaled) in models.items():

    # Pick scaled or unscaled data for this model.
    X_tr = X_train_scaled if uses_scaled else X_train
    X_v = X_val_scaled if uses_scaled else X_val

    # Train the model.
    model.fit(X_tr, y_train)

    # Predict on training and validation data.
    train_pred = model.predict(X_tr)
    val_pred = model.predict(X_v)

    # Train recall is used only for the overfitting check below.
    train_recall = recall_score(y_train, train_pred)
    val_recall = recall_score(y_val, val_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, val_pred),
        "Precision": precision_score(y_val, val_pred, zero_division=0),
        "Recall": val_recall,
        "F1-score": f1_score(y_val, val_pred),
        "Train Recall": train_recall,
        "Recall Gap (Train - Val)": train_recall - val_recall,
    })

    confusion_matrices[name] = confusion_matrix(y_val, val_pred)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(confusion_matrices[name])
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, val_pred, target_names=["No Failure", "Failure"], digits=3))


# -------------------- 5. Comparison Table --------------------

# Ranked by F1-score on the validation set (best first).
results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("MODEL COMPARISON (Validation Set, Failure Class)")
print("=" * 60)
print(results_df.round(3).to_string(index=False))

# Save the table so it can be used in the report / PPT.
results_df.round(3).to_csv("data/model_comparison.csv", index=False)


# -------------------- 6. Overfitting Check --------------------

# If a model scores much better on training data than on validation
# data, it has memorised the training set instead of learning general
# patterns (overfitting). A big "Recall Gap" is the warning sign.
#
# Why RECALL and not F1 or precision for this check:
# the training set is balanced (50% failures) but validation is not
# (~3.4% failures). Precision and F1 drop automatically when failures
# are rare, even for a perfect-fit model, so comparing them across the
# two sets would show a "gap" that is not overfitting at all.
# Recall only looks at the real failures ("how many did we catch?"),
# so it does not depend on the class balance and is a fair comparison.

print("\nOverfitting Check (bigger gap = more overfitting):")
print(results_df[["Model", "Train Recall", "Recall", "Recall Gap (Train - Val)"]].round(3).to_string(index=False))


# -------------------- 7. Confusion Matrix Plots --------------------

# One confusion matrix per model, in a 2 x 3 grid.
#   Top-left     = True Negatives  (healthy, predicted healthy)
#   Top-right    = False Positives (healthy, predicted failure -> false alarm)
#   Bottom-left  = False Negatives (failed, predicted healthy -> MISSED failure)
#   Bottom-right = True Positives  (failed, predicted failure -> caught)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, name in zip(axes.flatten(), results_df["Model"]):
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrices[name],
        display_labels=["No Failure", "Failure"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)

plt.suptitle("Confusion Matrices on Validation Set (ranked by F1-score)", fontsize=14)
plt.tight_layout()
plt.show()


# -------------------- 8. Metric Comparison Chart --------------------

# Bar chart of Precision, Recall and F1 for each model.

results_df.set_index("Model")[["Precision", "Recall", "F1-score"]].plot(
    kind="bar",
    figsize=(11, 5),
    color=["#4E6A9A", "#D2803A", "#A9443A"]
)
plt.title("Model Comparison on Validation Set (Failure Class)")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc="upper center", ncol=3)
plt.tight_layout()
plt.show()


# -------------------- Best Model --------------------

best = results_df.iloc[0]
print(f"\nBest model on validation (by F1-score): {best['Model']}")
print(f"Precision = {best['Precision']:.3f}, Recall = {best['Recall']:.3f}, F1 = {best['F1-score']:.3f}")

# Next steps (separate script):
#   - Tune the best model's settings (with cross-validation on the
#     original training data, SMOTENC applied inside each fold).
#   - Adjust the decision threshold.
#   - Evaluate ONCE on data/test.csv for the final reported score.
