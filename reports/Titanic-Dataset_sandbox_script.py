import matplotlib.pyplot as plt

# Compute the correlation matrix between numeric features
corr_matrix = X.corr()

# Print the top 3 most correlated pairs
print("Top 3 most correlated pairs:")
print(corr_matrix.nlargest(3, 'Fare'))

# Print the shape and basic statistics of X
print("\nShape of X:", X.shape)
print("Basic Statistics of X:\n", X.describe())

# Store a dictionary with key findings
metrics_output = {
    "Correlation Matrix": corr_matrix,
    "X Shape": X.shape,
    "X Basic Statistics": X.describe()
}

print("\nMetrics Output:")
for k, v in metrics_output.items():
    print(f"{k}: {v}")