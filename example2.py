import pandas as pd
import numpy as np
import random
import csv

usage_df = pd.read_csv('Data/usage_trends.csv')
usage_factors = ["House", "School", "Health_center", "Church", "Water_pump"]

dates = usage_df["Time"]

"""
0 = abnormal
1 = normal
"""

# Apply random zeroing across all usage factors at once
# Generate a random mask of the same shape as the DataFrame for the specified columns
mask = (np.random.rand(len(usage_df), len(usage_factors)) <= 0.01)

# Apply mask to zero out values
for idx, factor in enumerate(usage_factors):
    usage_df.loc[mask[:, idx], factor] = 0

usage_df["Class_name"] = 1
for factor in usage_factors:
    usage_df.loc[usage_df[factor] == 0, "Class_name"] = 0
usage_df.to_csv('Data/usage_traends_v2.csv')
print("done")