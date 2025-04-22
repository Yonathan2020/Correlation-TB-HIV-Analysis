# tb_analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configure plots to show in VS Code
#%matplotlib inline  

# 1. Load Data (adjust path if needed)
file_path = "TB_burden_countries2025-04-07.csv"
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully! Shape:", df.shape)
except Exception as e:
    print("Error loading file:", e)
    exit()

# 2. Missing Value Analysis
print("\n=== Missing Values ===")
print(df.isnull().sum())

# 3. Visualization
plt.figure(figsize=(12, 6))

# Before imputation
plt.subplot(1, 2, 1)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values (Original)")

# After imputation
plt.subplot(1, 2, 2)
df_filled = df.fillna(0)
sns.heatmap(df_filled.isnull(), cbar=False, cmap='viridis')
plt.title("After Filling NaN with 0")

plt.tight_layout()
plt.show()

# 4. Correlation Analysis
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()