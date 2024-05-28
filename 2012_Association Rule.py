#%% Import Files
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Read the CSV file
df_raw = pd.read_csv('df_bin.csv')

#%%
# Filter the data to include only instances where arrests were made
df_filtered = df_raw[df_raw['arstmade'] != 'Y'].copy()

#%%
# Drop rows with missing values
df_filtered.dropna(inplace=True)

#%%
# Encode the 'race' column into dummy variables
dummies = pd.get_dummies(df_filtered['race'])

#%%
# Include the 'age' column in the encoded dataframe
dummies['age'] = df_filtered['age']

#%%
# Remove duplicate rows
dummies.drop_duplicates(inplace=True)

#%%
# Group the data per instance
grouped_df = dummies.groupby('age').sum()

#%%
# Find frequent itemsets
frequent_itemsets = apriori(grouped_df, min_support=0.2, use_colnames=True)

#%%
# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

#%%
# Display the association rules
print(rules)

#%%
# Bar chart of frequent itemsets
plt.figure(figsize=(12, 6))
top_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(10)
sns.barplot(x='support', y='itemsets', data=top_itemsets)
plt.title('Top 10 Frequent Itemsets by Support')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.show()

#%%
# Create a scatter plot for Support vs Confidence
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.grid(True)
plt.show()