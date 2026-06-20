# Cell 1: Import core data science and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic style for visualizations
sns.set_theme(style="whitegrid")

# Cell 2: Load the Fragrantica raw dataset
df = pd.read_csv('data/raw_dataset.csv')
print(f"Dataset initial shape: {df.shape}")
display(df.head())

# Cell 3: Data Inspection & Cleaning Pipeline
print("--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# 1. Handle Numerical Columns (Impute missing ratings/votes with median to avoid outlier skew)
numerical_cols = ['Rating', 'Votes']
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 2. Handle Categorical Columns (Fill missing notes or gender labels with 'Shared/Unknown')
categorical_cols = ['Brand', 'Gender', 'Top_Notes', 'Middle_Notes', 'Base_Notes']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# 3. Outlier Mitigation (Cap extreme review counts using the Interquartile Range method)
if 'Votes' in df.columns:
    q1 = df['Votes'].quantile(0.25)
    q3 = df['Votes'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    # Cap outliers at the upper limit instead of dropping rows to preserve dataset size
    df.loc[df['Votes'] > upper_limit, 'Votes'] = upper_limit

# 4. Export the cleaned pipeline data
df.to_csv('data/cleaned_dataset.csv', index=False)
print("\nData pipeline cleaning completed successfully. Cleaned dataset saved.")

# Cell 4: Descriptive Statistical Analysis
display(df.describe())

# Cell 5: Visualization 1 - Distribution of Fragrantica User Ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], bins=25, kde=True, color='darkslateblue')
plt.title('Distribution of Community Ratings on Fragrantica')
plt.xlabel('User Rating (Out of 5)')
plt.ylabel('Count of Fragrances')
plt.savefig('images/rating_distribution.png', dpi=300)
plt.show()

# Cell 6: Visualization 2 - Scatter Plot (Popularity vs. Rating)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Votes', y='Rating', hue='Gender', data=df, alpha=0.6)
plt.title('Fragrance Popularity (Votes) vs. Overall Community Rating')
plt.xlabel('Number of Votes (Capped)')
plt.ylabel('Average Rating')
plt.savefig('images/votes_vs_rating.png', dpi=300)
plt.show()

# Cell 7: Visualization 3 - Correlation Heatmap
plt.figure(figsize=(8, 6))
# Filter strictly for numerical metrics to avoid datatype mapping warnings
numerical_df = df.select_dtypes(include=[np.number])
correlation = numerical_df.corr()

sns.heatmap(correlation, annot=True, cmap='mako', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Fragrantica Numerical Fields')
plt.savefig('images/correlation_matrix.png', dpi=300)
plt.show()