"""
Quick check that counts the missing values in each column of between_channel_features.csv.

Needs: between_channel_features.csv (from between_channel_features.py).
"""
import pandas as pd
import os
csv = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "01_data_and_features", "between_channel and temp explore", "between_channel_features", "between_channel_features.csv"))
df= pd.read_csv(csv)
print(df.isna().sum())