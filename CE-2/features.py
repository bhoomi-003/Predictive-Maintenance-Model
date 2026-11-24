# ==================================================================
# FEATURE ENGINEERING (NEW FILE)
#
# Creates 3 new features from the existing sensor readings.
# Kept in its own file so datacleaning.py and modeltuning.py use
# exactly the same formulas, and so the saved model can be loaded
# later (the model needs this function to exist when it is loaded).
#
# Why these features:
# Each failure type in the AI4I 2020 dataset is linked to a physical
# quantity that is not a column on its own, but can be calculated
# from columns we already have:
#   Heat dissipation failure -> temperature difference is too small
#   Power failure            -> mechanical power is too low or too high
#   Overstrain failure       -> torque x tool wear is too high
#
# This is NOT data leakage: every new value is calculated only from
# the sensor readings in the same row, never from "Machine failure".
# ==================================================================

import numpy as np


def add_features(X):
    X = X.copy()

    # Process temperature minus air temperature (in Kelvin).
    # A small difference means heat is not being carried away well.
    X["Temp difference [K]"] = X["Process temperature [K]"] - X["Air temperature [K]"]

    # Mechanical power in watts: torque x angular speed.
    # Angular speed (rad/s) = rpm x 2 x pi / 60.
    X["Power [W]"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60

    # Strain on the tool: high torque on a worn tool means overstrain.
    X["Torque x Tool wear"] = X["Torque [Nm]"] * X["Tool wear [min]"]

    # Round to 2 decimals to remove floating-point noise
    # (e.g. 10.799999999 -> 10.8). Does not change the information.
    for column in ENGINEERED_COLUMNS:
        X[column] = X[column].round(2)

    return X


# Names of the new columns (used to drop / select them in other files).
ENGINEERED_COLUMNS = ["Temp difference [K]", "Power [W]", "Torque x Tool wear"]
