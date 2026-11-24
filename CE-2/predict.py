# ==================================================================
# PREDICTIVE MAINTENANCE SYSTEM: PREDICT FOR NEW MACHINES (NEW FILE)
#
# Loads the final Random Forest model saved by modeltuning.py and
# checks whether a machine is at risk of failure from its sensor
# readings. This is how the system would be used in a factory:
# new readings come in -> the model gives a failure probability ->
# if it is above the threshold, the machine is flagged for maintenance.
#
# Needs: models/random_forest_final.pkl (made by modeltuning.py)
#        features.py in the same folder (the model uses add_features)
#
# Run order: ... -> modeltuning.py -> predict.py
# ==================================================================

import joblib
import pandas as pd

# The saved model calls add_features() from features.py, so this file
# must be importable (same folder) before the model is loaded.
import features  # noqa: F401


# -------------------- 1. Load the Saved Model --------------------

saved = joblib.load("models/random_forest_final.pkl")
model = saved["model"]          # pipeline: features -> encoding -> Random Forest
threshold = saved["threshold"]  # decision threshold chosen on the validation set (0.75)

print("Model loaded. Decision threshold:", threshold)


# -------------------- 2. Prediction Function --------------------

def predict_machine(air_temp, process_temp, rpm, torque, tool_wear, machine_type):
    """
    Takes the 5 raw sensor readings + machine type and returns
    the failure probability and whether the machine is at risk.

    The 3 engineered features and the one-hot encoding of Type are
    created automatically inside the saved pipeline, so only the
    original readings are needed here.
    """
    reading = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": machine_type,
    }])

    probability = model.predict_proba(reading)[0, 1]   # probability of failure
    at_risk = probability >= threshold
    return probability, at_risk


def show_result(name, probability, at_risk):
    status = "AT RISK - schedule maintenance" if at_risk else "Healthy - no action needed"
    print(f"{name:<36} failure probability = {probability:6.1%}   ->  {status}")


# -------------------- 3. Example Machines --------------------

# Example readings to demonstrate the system. Values are within the
# normal ranges of the dataset (see df.describe() in datapreprocessing.py).
examples = [
    # name,                       air,   process, rpm,  torque, wear, type
    ("Machine A (normal operation)", 298.5, 308.7, 1550, 38.0, 60, "M"),
    ("Machine B (worn tool, high torque)", 299.0, 309.5, 1320, 64.0, 215, "L"),
    ("Machine C (poor heat removal)", 302.5, 310.5, 1290, 55.0, 90, "L"),
    ("Machine D (high speed, low torque)", 298.0, 308.0, 2800, 4.5, 30, "H"),
    ("Machine E (new tool, medium load)", 300.2, 310.0, 1480, 42.0, 10, "H"),
]

print("\n" + "=" * 100)
print("PREDICTIONS FOR EXAMPLE MACHINES")
print("=" * 100)
for name, air, process, rpm, torque, wear, mtype in examples:
    probability, at_risk = predict_machine(air, process, rpm, torque, wear, mtype)
    show_result(name, probability, at_risk)


# -------------------- 4. Enter Your Own Reading (optional) --------------------

# Lets the user type in readings for any machine, e.g. during a demo.

answer = input("\nDo you want to check your own machine reading? (y/n): ").strip().lower()

while answer == "y":
    try:
        air = float(input("  Air temperature [K]     (e.g. 300): "))
        process = float(input("  Process temperature [K] (e.g. 310): "))
        rpm = float(input("  Rotational speed [rpm]  (e.g. 1500): "))
        torque = float(input("  Torque [Nm]             (e.g. 40): "))
        wear = float(input("  Tool wear [min]         (e.g. 100): "))
        mtype = input("  Machine type (L / M / H): ").strip().upper()

        if mtype not in ["L", "M", "H"]:
            print("  Machine type must be L, M or H. Try again.\n")
            continue

        probability, at_risk = predict_machine(air, process, rpm, torque, wear, mtype)
        print()
        show_result("Your machine", probability, at_risk)

    except ValueError:
        print("  Please enter numbers only for the sensor readings. Try again.\n")
        continue

    answer = input("\nCheck another machine? (y/n): ").strip().lower()

print("\nDone.")
