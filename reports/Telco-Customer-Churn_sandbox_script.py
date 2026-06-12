import pandas as pd
import numpy as np

metrics_output = {}

# 1. Compute correlation matrix between numeric features and print top 3 most correlated pairs
correlation_matrix = X.corr()
print("Top 3 most correlated pairs:")
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            print(f"{correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]}")

# 2. Print shape and basic statistics of X
print("Shape of X:", X.shape)
print("Basic statistics of X:")
print(X.describe())

metrics_output = {"Correlation Matrix": correlation_matrix,
                  "X Shape": X.shape,
                  "X Basic Statistics": X.describe()}