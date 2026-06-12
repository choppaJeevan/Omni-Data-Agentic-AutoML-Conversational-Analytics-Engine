import matplotlib.pyplot as plt

plt.hist(df['Survived'], bins=2, edgecolor='black')
plt.xlabel('Survival Status')
plt.ylabel('Frequency')
plt.title('Distribution of Survived vs. Perished')
plt.savefig('plots/Titanic-Dataset_survival_distribution.png')