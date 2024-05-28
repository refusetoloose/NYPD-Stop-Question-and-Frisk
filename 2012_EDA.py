#%%
#Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#%%
# Load the data set
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 150)
df_raw= pd.read_csv('2012.csv')
df_raw.head()

# Get meaning and type of data
# Create a string to store data types for all columns
data_types_string = ""

# Concatenate column names and data types
for column in df_raw.columns:
    data_types_string += f"{column}: {df_raw[column].dtype}, "

# Print the string
print(data_types_string)


#%%
# Get meaning and type of data
df_raw.info()
df_raw.dtypes
df_raw.describe(include='all')

# %% Verify data quality
# Are there Duplicate data?
df_raw[df_raw.duplicated()].shape

# %%
# Are there missing values?
df_raw.isnull().sum()

# %% How do you deal with these problems?
#Remove all the rows with null value and generate stats again
df_clean = df_raw.dropna()
df_clean.shape

#%% Check How Many rows were removed
print(len(df_raw) - len(df_clean))

#%% Generate Stats again
df_clean.isnull().sum()
#%%
# Describe categorical columns
categorical_columns = df_clean.select_dtypes(include=['object'])
categorical_description = categorical_columns.describe(include='all')
print("Categorical Columns Description:\n", categorical_description)
categorical_columns.columns.shape

#%%
# Describe numerical columns
numerical_columns = df_clean.select_dtypes(exclude=['object'])
numerical_description = numerical_columns.describe(include='all')
print("Numerical Columns Description:\n", numerical_description)
numerical_description.columns.shape

#%% Are there any Outliers?
# Lets Find out using Zscores 
from scipy.stats import zscore
# Calculate Z-scores for numerical columns
z_scores = df_clean.select_dtypes(include='number').apply(zscore)
# Identify outliers based on a threshold (e.g., 3 or -3)
outliers = (z_scores > 3) | (z_scores < -3)
# Count the number of outliers in each numerical column
num_outliers = outliers.sum()
# Print the number of outliers for each numerical column
print("Number of outliers in each numerical column:")
print(num_outliers)
#Yes based on Zscores, ser_num and perobs age and ht_feet and weight seems to be outliers
#Lets ignore ser_num as it doesnt add much value

#%% Visualize distribution of identified outliers

# Columns with outliers identified using z-score
outlier_columns_zscore = ['perobs', 'age', 'ht_feet', 'weight']

# Create box plots for each column
for column in outlier_columns_zscore:
    plt.figure(figsize=(8, 6))
    plt.boxplot(df_clean[column], vert=False)
    plt.title(f'Box plot of {column} with outliers (z-score method)')
    plt.xlabel(column)
    plt.show()

#%%
df_clean['age'].unique()

#%%
len(df_clean[(df_clean['age'] >= 5) & (df_clean['age'] <= 100)]), len(df_clean[(df_clean['age'] < 5) | (df_clean['age'] > 100)])

#%%Lets filter out age 
mode_age = df_clean['age'].mode()[0]
df_clean.loc[(df_clean['age'] < 5) | (df_clean['age'] > 100), 'age'] = mode_age

#%%
df_clean['age'].unique()

#%%
df_clean['weight'].unique()
#%%
len(df_clean[(df_clean['weight'] <= 500) & (df_clean['weight'] >= 20)]), len(df_clean[(df_clean['weight'] > 500) | (df_clean['weight'] < 20)])
#%%
mode_weight = df_clean['weight'].mode()[0]
df_clean.loc[(df_clean['weight'] > 500) | (df_clean['weight'] < 20), 'weight'] = mode_weight

#%% Lets Keep the Height in one column
# Lets convert the height in feet to inches and keep the height value of inches in one column
df_clean['height'] = df_clean['ht_feet'] * 12 + (df_clean['ht_inch'])

# Drop ht_feet and ht_inch columns
df_clean.drop(columns=['ht_feet', 'ht_inch'], inplace=True)

#%% Lets handle the perobs
df_clean['perobs'].dtype
#%%
# Replace '**' with the mode of 'perobs'
mode_perobs = df_clean['perobs'].mode()[0]
df_clean['perobs'] = df_clean['perobs'].apply(lambda x: mode_perobs if x == '**' else x)

# Convert 'perobs' to numeric
df_clean['perobs'] = pd.to_numeric(df_clean['perobs'], errors='coerce').fillna(60)

# Convert 'perobs' to integer
df_clean['perobs'] = df_clean['perobs'].astype(int)

# Limit 'perobs' to a maximum of 60 minutes
df_clean['perobs'] = df_clean['perobs'].apply(lambda x: min(x, 60))

#%% Visualization for Distribution of Age
plt.figure(figsize=(10, 6))
plt.hist(df_clean['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%% Visualization for Distibution of Weight
plt.figure(figsize=(10, 6))
plt.hist(df_clean['weight'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Weight')
plt.xlabel('Weight')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%% Visualize Length of observation before Engaging
plt.figure(figsize=(10, 6))
plt.hist(df_clean['perobs'], bins=20, color='skyblue', edgecolor='black')
plt.title('Length of Observation Before Engaging')
plt.xlabel('Perobs')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%% Lets categorize age and plot to see which age group were more likey suspected
labels = ['child', 'teen', 'young_adult', 'adult', 'middle_aged', 'older', 'senior']
points = [0, 12, 18, 25, 35, 55, 70, 100]
df_clean.loc[:, 'age_category'] = pd.cut(df_clean['age'], bins=points, labels=labels, right=False)

# %%
plt.figure(figsize=(10, 6))
df_clean['age_category'].value_counts().plot(kind='bar')
plt.title('Distribution of Age Categories')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
#%%
df_clean[['age','height','weight','perobs']].describe().transpose()
#%%
## Lets do data cleaning and handle unknown values
df_clean.forceuse.value_counts()

#%% Remove replace blank with other in the 'forceuse' column
df_clean['forceuse'].unique()
df_clean['forceuse'].replace('', 'OT', inplace=True)
force_use_mapping = {
    'DS': 'DEFENSE OF SELF',
    'DO': 'DEFENSE OF OTHER',
    'OR': 'OVERCOME RESISTANCE',
    'OT': 'OTHER',
    'SF': 'SUSPECTED FLIGHT',
    'SW': 'SUSPECTED WEAPON'
}
df_clean['forceuse'] = df_clean['forceuse'].map(force_use_mapping)

#%% Lets not remove sex as many doesnt identify themself in both
df_clean.sex.value_counts()
# df_clean['sex'].unique()
# df_clean = df_clean[~df_clean['sex'].isin(['Z'])]
sex_mapping = {'M': 'Male', 'F': 'Female'}
df_clean['sex'] = df_clean['sex'].map(sex_mapping).fillna('Other')

#%% Lets Merge 'U' Unknown and 'Z' Other into a single category 'Z' Other
df_clean['race'].unique()
df_clean['race'].replace(['Z', 'U'], 'Z', inplace=True)
race_mapping = {
    'B': 'BLACK',
    'Q': 'WHITE-HISPANIC',
    'W': 'WHITE',
    'P': 'BLACK-HISPANIC',
    'A': 'ASIAN/PACIFIC ISLANDER',
    'Z': 'OTHER',
    'U': 'UNKNOWN',
    'I': 'AMERICAN INDIAN/ALASKAN NATIVE'
}
df_clean['race'] = df_clean['race'].map(race_mapping).fillna(np.nan)

#%%
df_clean['haircolr'].unique()
df_clean['haircolr'].replace('XX', 'ZZ', inplace=True)
hair_color_mapping = {
    'BA': 'BALD',
    'BK': 'BLACK',
    'BL': 'BLOND',
    'BR': 'BROWN',
    'DY': 'DYED',
    'FR': 'FROSTED',
    'GY': 'GRAY',
    'RA': 'RED',
    'SN': 'SANDY',
    'SP': 'SALT AND PEPPER',
    'WH': 'WHITE',
    'XX': 'UNKNOWN',
    'ZZ': 'OTHER'
}
df_clean['haircolr'] = df_clean['haircolr'].map(hair_color_mapping)
#%%
df_clean['eyecolor'].unique()
df_clean['eyecolor'].replace(['XX', 'MC'], 'Z', inplace=True)
eye_color_mapping = {
    'BK': 'BLACK',
    'BL': 'BLUE',
    'BR': 'BROWN',
    'DF': 'TWO DIFFERENT',
    'GR': 'GREEN',
    'GY': 'GRAY',
    'HA': 'HAZEL',
    'MA': 'MAROON',
    'P': 'PINK',
    'VI': 'VIOLET',
    'XX': 'UNKNOWN',
    'Z': 'OTHER'
}
df_clean['eyecolor'] = df_clean['eyecolor'].map(eye_color_mapping)

#%%
df_clean['build'].unique()
df_clean = df_clean[~df_clean['build'].isin(['Z'])]
build_mapping = {
    'H': 'HEAVY',
    'M': 'MEDIUM',
    'T': 'THIN',
    'U': 'MUSCULAR',
    'Z': 'UNKNOWN'
}
df_clean['build'] = df_clean['build'].map(build_mapping)

#%%
#Visualize distribution of race and city
plt.subplot(121)
df_clean.groupby('race').size().plot.pie(colormap='coolwarm')
plt.title('Distribution of Stopped Individuals by Race')
plt.subplot(122)
df_clean.groupby('city').size().plot.pie(colormap='viridis')
plt.title('Distribution of Stops by City')
plt.subplots_adjust(left=0.0, bottom=0.0, right=2, top=1.1, wspace=0.01, hspace=0.2)
plt.show()

#%% Visualize frisks by race
sns.catplot(x='race', hue='frisked', data=df_clean, kind='count', palette='coolwarm', edgecolor='0.6')
plt.title('Frisks by Race')
plt.grid()
plt.subplots_adjust(left=0.0, bottom=0.0, right=2, top=0.8, wspace=0.01, hspace=0.2)

# %%
# Visualize distribution of frisks or no frisks by time of day
df_clean.pivot(columns='frisked')['timestop'].plot(kind='hist', stacked=True, bins=30, alpha=0.9, edgecolor='black')
plt.xticks(np.arange(0, 2500, step=600))
plt.xlim([0, 2400])
plt.xlabel('Time of Stop [HH:MM]')
plt.title('Distribution of Frisks by Time of Day')
plt.legend(['Not Frisked', 'Frisked'], loc='upper right')  
plt.show()

# %%
# Visualize frisks by sex
sns.catplot(x='sex', hue='frisked', data=df_clean, kind='count', palette='coolwarm', edgecolor='0.6')
plt.title('Frisks by Sex'); plt.grid()
plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=0.6, wspace=0.01, hspace=0.2)

# %%
# Visualize frisks by city
sns.catplot(x='city', hue='frisked', data=df_clean, kind='count', palette='coolwarm', edgecolor='0.6')
plt.title('Frisks by city'); plt.grid()
plt.subplots_adjust(left=0.0, bottom=0.0, right=2, top=0.8, wspace=0.01, hspace=0.2)

#%%
# Scatter plot of age, height, and weight
sns.pairplot(df_clean[['age', 'height', 'weight']])
plt.show()

# Correlation matrix
corr_matrix = df_clean[['age', 'height', 'weight']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Cross-tabulation between reasons for SQF and type of force used
cross_tab = pd.crosstab(df_clean['crimsusp'], df_clean['forceuse'])
print(cross_tab)

# Group-wise averages of numerical attributes by reasons for SQF and type of force used
grouped_avg = df_clean.groupby(['crimsusp', 'forceuse'])[['age', 'height', 'weight']].mean()
print(grouped_avg)


#%%
# Group-wise averages of numerical attributes by crimsusp, forceuse, and type of force used
grouped_avg_with_force = df_clean.groupby(['crimsusp', 'forceuse', 'pf_hands', 'pf_wall', 'pf_grnd', 'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_pepsp', 'pf_other'])[['age', 'height', 'weight']].mean()
print(grouped_avg_with_force)

# %%
#Creating a column to indicate whether someone was arrested, summonsed, or neither
arrest_condition = df_clean['arstmade'] == 'Y'
summons_condition = df_clean['sumissue'] == 'Y'
df_clean.loc[arrest_condition, 'outcome'] = 'Arrested'
df_clean.loc[summons_condition, 'outcome'] = 'Summons Issued'
sns.countplot(x='outcome', data=df_clean, palette='Set2')
plt.title('Outcomes of Police Encounters')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

#%%
df_clean['trhsloc'].unique()
df_clean.dropna(subset=['trhsloc'], inplace=True)
# %%
# re-classifying trhsloc
Location_mapping = {
    'P': 'Neither',
    'H': 'Housing',
    'T': 'Transit'
}
df_clean['trhsloc'] = df_clean['trhsloc'].map(Location_mapping)
#%%
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, x='trhsloc', palette='coolwarm', edgecolor='0.6')
plt.title('Distribution of Locations')
plt.xlabel('Location')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%%
df_clean['crimsusp'].value_counts()[0:5]

#%%
df_clean.head()

# %%
cross_tab = pd.crosstab(df_clean['cs_descr'], df_clean['forceuse'])

# Plotting cross-tabulation
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, cmap='Blues', annot=True, fmt='d', cbar=True)
plt.title('Cross-tabulation of Reason for SQF vs. Type of Force Used')
plt.xlabel('Type of Force Used')
plt.ylabel('Reason for SQF')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%Group-wise averages
grouped_data = df_clean.groupby('cs_descr')['forceuse'].value_counts(normalize=True).unstack()

# Plotting group-wise averages
plt.figure(figsize=(12, 6))
grouped_data.plot(kind='bar', stacked=True, cmap='tab10')
plt.title('Group-wise Averages of Type of Force Used by Reason for SQF')
plt.xlabel('Reason for SQF')
plt.ylabel('Proportion of Type of Force Used')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Type of Force Used', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
df_clean.head()
#%%
df_clean = df_clean.drop(['outcome', 'age_category'], axis=1)
#%%
#Export Cleaned data to csv
df_clean.to_csv('df_clean.csv', index=False)
#%%
# Define the columns containing 'Y' and 'N' values
columns_to_convert = [
    'othpers', 'arstmade', 'sumissue', 'offunif', 'frisked', 'searched', 'contrabn', 'adtlrept', 
    'pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap', 'pf_hands', 'pf_wall', 
    'pf_grnd', 'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_pepsp', 'pf_other', 'radio', 
    'ac_rept', 'ac_inves', 'rf_vcrim', 'rf_othsw', 'ac_proxm', 'rf_attir', 'cs_objcs', 'cs_descr', 
    'cs_casng', 'cs_lkout', 'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc', 'cs_furtv', 
    'rf_rfcmp', 'ac_cgdir', 'rf_verbl', 'cs_vcrim', 'cs_bulge', 'cs_other', 'ac_incid', 'ac_time', 
    'rf_knowl', 'ac_stsnd', 'ac_other', 'sb_hdobj', 'sb_outln', 'sb_admis', 'sb_other', 'rf_furt', 
    'rf_bulg'
]

# Convert 'Y' and 'N' to binary (1 and 0) for the specified columns and change the data type to int
df_clean[columns_to_convert] = df_clean[columns_to_convert].replace({'Y': 1, 'N': 0}).astype(int)

#Export cleaned binary converted data to csv
df_clean.to_csv('df_bin.csv', index=False)

# Now 'Y' and 'N' values in specified columns are replaced with 1 and 0, respectively, and data type is changed to int
#%%
numerical_columns = df_clean.select_dtypes(exclude=['object'])
numerical_description = numerical_columns.describe(include='all')
print("Numerical Columns Description:\n", numerical_description)
numerical_description.columns.shape


# %%
# Pickle Numerical Columns
numerical_columns.to_pickle("numerical_columns.pkl")

# %%
