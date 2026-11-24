# ==================================================================
# CHANGES FROM THE CE1 VERSION OF THIS FILE
#
# [CHANGE 1] Loads data/ai4i2020_modified.csv instead of the original
#            ai4i2020.csv (the modified copy has missing values and
#            duplicate rows - see create_modified_dataset.py).
# [CHANGE 2] The exploration checks (head, shape, columns, info,
#            describe, failure distribution) are active again instead
#            of commented out, so the problems in the data are shown.
# [CHANGE 3] NEW: missing-value and duplicate checks added.
#            This file only FINDS the problems; datacleaning.py FIXES
#            them (duplicates removed, missing values filled).
# [CHANGE 4] The grouped averages are now calculated on a copy with
#            duplicates removed, so repeated rows are not counted twice.
#
# Unchanged from CE1: the comparison of failed vs not-failed machines.
# ==================================================================

import pandas as pd  # pandas library for working on tables and datasets.


# Data Exploration and Preprocessing

# [CHANGE 1] CE1 loaded "data/ai4i2020.csv".
df = pd.read_csv("data/ai4i2020_modified.csv")  # df is a dataframe, which is like a table in Excel.

# [CHANGE 2] The checks below were commented out in CE1. They are
# active again so the output shows the state of the modified dataset.

print(df.head())  # prints first 5 rows of the dataset(checks data loads correctly or not).

print("\nDataset Shape:")
print(df.shape) # gives us the number of rows and columns in the dataset.
# [CHANGE 2] Now 10150 rows instead of 10000 (150 duplicate rows were added).

print("\nColumn Names:")
print(df.columns) # prints the names of the columns in the dataset.

print("\nDataset Information:")
df.info()  # It answers how much data do we have, data type of each column, and are there any missing values in the dataset.
# [CHANGE 2] The "Non-Null Count" is now less than 10150 for the six
# feature columns, which shows they contain missing values.

print("\nStatistical Summary:")
print(df.describe()) # For numerical columns, it gives us the count, mean, standard deviation, minimum, maximum, and quartiles.
# The min/max ranges are the same as in CE1, so there are still no
# inconsistent / out-of-range values. "count" is lower where values are missing.


# -------------------- [CHANGE 3] NEW: Missing Values Check --------------------

print("\nMissing Values per Column:")
print(df.isnull().sum())  # counts the empty (NaN) cells in each column.

print("\nTotal Missing Values:", df.isnull().sum().sum())
print("Rows with at least one Missing Value:", df.isnull().any(axis=1).sum())
# Result: about 100 missing values in each of Type, Air temperature,
# Process temperature, Rotational speed, Torque and Tool wear
# (605 in total, spread over 596 rows). No missing values in the target.


# -------------------- [CHANGE 3] NEW: Duplicate Rows Check --------------------

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())  # counts rows that are an exact copy of an earlier row.
# Result: 150 duplicate rows.


print("\nMachine Failure Distribution:")
print(df["Machine failure"].value_counts()) # gives us the count of each unique value in the "Machine failure" column, which helps us understand the distribution of machine failures in the dataset.

# [CHANGE 2] With duplicates included: 9808 not failed, 342 failed.
# After removing duplicates it is the same as CE1: 9661 not failed, 339 failed.
# This indicates that the dataset is imbalanced, with a majority of machines not failing.
# The imbalance is handled in datacleaning.py (SMOTENC on the training set only).


# [CHANGE 4] Remove duplicates in a separate copy for the comparison below,
# so repeated rows do not affect the averages. Missing values are skipped
# automatically by mean(). The original df is not changed here.
df_unique = df.drop_duplicates()

# Differences between failed and not failed machines.
print("\nAverage Values Grouped by Machine Failure:")
print(
    df_unique.groupby("Machine failure")[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    ].mean()
) # i found out that average of all features of machines which failed is higher, rs is lower than not failed machines.
# this observation is not fully correct right now, but we will explore more in the next steps of data preprocessing and feature engineering.
# [CHANGE 4] The averages differ from CE1 only slightly (because of the
# missing values), so the observation above still holds.
