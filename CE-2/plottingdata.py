# ==================================================================
# CHANGES FROM THE CE1 VERSION OF THIS FILE
#
# [CHANGE 1] Loads data/ai4i2020_modified.csv instead of the original
#            ai4i2020.csv.
# [CHANGE 2] NEW: duplicate rows are removed straight after loading,
#            so repeated rows are not counted twice in the plots and
#            failure rates.
#
# Missing values need no extra code here: seaborn leaves out empty
# cells when drawing box plots, and value_counts() / groupby() skip
# them when counting.
#
# Unchanged from CE1: all plots, failure-rate calculations and
# observations. With the modified data, the numbers change only
# slightly and the conclusions stay the same.
# ==================================================================

import pandas as pd
import matplotlib.pyplot as plt #Matplotlib is a Python library used to create graphs and charts.
import seaborn as sns # it makes matplotlib plots easier to create and more visually appealing.

# [CHANGE 1] CE1 loaded "data/ai4i2020.csv".
df = pd.read_csv("data/ai4i2020_modified.csv")

# [CHANGE 2] NEW: remove the 150 duplicate rows before plotting.
df = df.drop_duplicates()

# ---------------Visualizing rotational speed for failed and non-failed machines

# sns.boxplot(
#     data=df,
#     x="Machine failure",
#     y="Rotational speed [rpm]"
# )

# plt.title("Rotational Speed vs Machine Failure")
# plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
# plt.ylabel("Rotational Speed (RPM)")

# plt.show() # The central 50% of failed and non-failed observations show very little or no overlap in rotational speed,
# # but the complete groups still overlap when considering all observations.

# --------------- Visualizing torque for failed and non-failed machines
# sns.boxplot(
#     data=df,
#     x="Machine failure",
#     y="Torque [Nm]"
# )

# plt.title("Torque vs Machine Failure")
# plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
# plt.ylabel("Torque (Nm)")

# plt.show()

# Failed machines tend to operate at significantly higher torque values compared with non-failed machines in this dataset.
# Some failed machines with very low torque (the dots below the lower whisker).
# #Some non-failed machines with high torque (dots above ~66 Nm).



# --------------- Visualizing tool wear for failed and non-failed machines

# sns.boxplot(
#     data=df,
#     x="Machine failure",
#     y="Tool wear [min]"
# )

# plt.title("Tool Wear vs Machine Failure")
# plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
# plt.ylabel("Tool Wear (min)")

# plt.show()

# Failed machines tend to operate at significantly higher tool wear values compared with non-failed machines in this dataset.
# But there is major overlap between the two groups, therefore it alone is not a good predictor of machine failure.
# Combining it with other features may improve model prediction performance.


#--------------- Visualizing Air temperature for failed and non-failed machines

# sns.boxplot(
#     data=df,
#     x="Machine failure",
#     y="Air temperature [K]"
# )

# plt.title("Air Temperature vs Machine Failure")
# plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
# plt.ylabel("Air Temperature (K)")

# plt.show()

# Air temperature shows a slight difference between failed and non-failed machines, but by itself it does not appear to clearly separate the two groups.


# --------------- Visualizing Process temperature for failed and non-failed machines

# sns.boxplot(
#     data=df,
#     x="Machine failure",
#     y="Process temperature [K]"
# )

# plt.title("Process Temperature vs Machine Failure")
# plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
# plt.ylabel("Process Temperature (K)")

# plt.show()

# Process temperature appears slightly higher for failed machines, 
# but there is substantial overlap, suggesting that process temperature alone is not a strong indicator of machine failure.

print("\nMachine Type Distribution:")
print(df["Type"].value_counts()) # gets total number of each machine type in the dataset.

print("\nMachine Failures by Type:")
print(df.groupby("Type")["Machine failure"].sum()) # Number of machines failed by type.


print("\nFailure Rate by Machine Type:")
print(df.groupby("Type")["Machine failure"].mean() * 100) # L -> M -> H failure rate.

# --------------- [CHANGE 4] NEW: Bar chart of class distribution
# Draws the "Class Distribution: Machine Failure" chart used on slide 9 of the PPT.
# Uses the modified dataset with duplicates removed (done at the top of this file),
# so the counts match the original data: 9661 not failed, 339 failed.

class_counts = df["Machine failure"].value_counts().sort_index()  # 0 first, then 1

plt.figure(figsize=(6, 4.5))
bars = plt.bar(
    ["No Failure (0)", "Failure (1)"],
    class_counts.values,
    color=["#4E6A9A", "#A9443A"]
)

# Write the count on top of each bar.
for bar, value in zip(bars, class_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 100,
        f"{value}",
        ha="center",
        fontsize=10
    )

plt.title("Class Distribution: Machine Failure")
plt.ylabel("Number of Records")
plt.ylim(0, 10500)
plt.tight_layout()

plt.show()

# Only 339 of 10,000 records (3.39%) are failures -> the dataset is heavily imbalanced.
# This is handled later in datacleaning.py with SMOTENC (training set only).

# --------------- [CHANGE 3] NEW: Bar chart of failure rate by machine type
# Draws the "Failure Rate by Machine Type" chart used on slide 9 of the PPT.
# Uses the modified dataset with duplicates removed (done at the top of this file).
# Rows where Type is missing are left out automatically by groupby().

failure_rate = df.groupby("Type")["Machine failure"].mean() * 100
failure_rate = failure_rate.reindex(["L", "M", "H"])  # show bars in L -> M -> H order

plt.figure(figsize=(6, 4.5))
bars = plt.bar(
    failure_rate.index,
    failure_rate.values,
    color=["#A9443A", "#D2803A", "#4E6A9A"]
)

# Write the percentage on top of each bar.
for bar, value in zip(bars, failure_rate.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.05,
        f"{value:.2f}%",
        ha="center",
        fontsize=10
    )

plt.title("Failure Rate by Machine Type")
plt.xlabel("Machine Type (L = Low, M = Medium, H = High Quality)")
plt.ylabel("Failure Rate (%)")
plt.ylim(0, 4.5)
plt.tight_layout()

plt.show()

# L 3.91%, M 2.76%, H 2.11% -> low-quality machines fail most often,
# high-quality machines least often (same pattern as CE1).
# With the modified data: L 3.91%, M 2.76%, H 2.11% (CE1: 3.92%, 2.77%, 2.09%).
# Rows whose Type is missing are not counted in any group, which is
# why the rates change slightly. The order L > M > H is unchanged.
# L is low quality machines, M is medium quality machines, and H is high quality machines. 
# The failure rate is highest for low quality machines, followed by medium quality machines, and lowest for high quality machines.
