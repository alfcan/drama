import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest

# Load dataset
df = pd.read_csv('../../results/results.csv')

# Descriptive statistics
desc = df.describe()
desc.to_csv('../../results/descriptive_statistics.csv')


# Box plot for pre-mutation symptoms
plt.figure(figsize=(12, 6))
df.filter(like='pre_').boxplot()
plt.title('Box plot for pre-mutation symptoms')
plt.xticks(rotation=90)
plt.show()

# Box plot for post-mutation symptoms
plt.figure(figsize=(12, 6))
df.filter(like='post_').boxplot()
plt.title('Box plot for post-mutation symptoms')
plt.xticks(rotation=90)
plt.show()

# Perform Shapiro-Wilk test for normality
for column in df.columns:
    if 'pre_' in column or 'post_' in column:
        stat, p_value = shapiro(df[column])
        print(f'Shapiro-Wilk Test for {column}: Statistics={stat}, p-value={p_value}')
        if p_value > 0.05:
            print(f'{column} follows a normal distribution')
        else:
            print(f'{column} does not follow a normal distribution')
