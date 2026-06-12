import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df, X, y are already defined with the dataset

# Plot 1: Distribution of Churn
plt.figure(figsize=(8, 6))
sns.countplot(x="Churn", data=df)
plt.title("Distribution of Churn")
plt.xlabel("Churn Status")
plt.ylabel("Count")
plt.savefig('plots/Telco-Customer-Churn_churn_distribution.png')

# Plot 2: Correlation Matrix (top 5 features)
corr_matrix = X.corr().abs()
top_5_features = ['tenure', 'MonthlyCharges', 'Contract_Two year', 'InternetService_Fiber optic', 'PaymentMethod_Electronic check']
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix[top_5_features].corr(), annot=True, cmap='coolwarm', square=True)
plt.title("Correlation Matrix (Top 5 Features)")
plt.xlabel("Features")
plt.ylabel("Features")
plt.savefig('plots/Telco-Customer-Churn_correlation_matrix.png')

# Plot 3: Feature Importance
importances = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
sns.barplot(x="Feature", y="Importance", data=importances)
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.savefig('plots/Telco-Customer-Churn_feature_importances.png')