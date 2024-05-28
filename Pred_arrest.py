#%%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df_clean= pd.read_csv('df_bin.csv')
#%%
# Drop rows with missing values in the target variable 'arstmade'
df_clean_arstmade = df_clean.dropna(subset=['arstmade'])

#%%
# Select features and target variable
X = df_clean_arstmade[['frisked', 'searched', 'contrabn', 'pf_hands', 'pf_wall', 'pf_grnd',
                       'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_pepsp', 'pf_other']]
y = df_clean_arstmade['arstmade']

#%%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# One-hot encode the 'forceuse' variable
df_clean = pd.get_dummies(df_clean, columns=['forceuse'], drop_first=True)

#%%
# Initialize and train the model
model_arstmade = LogisticRegression()
model_arstmade.fit(X_train, y_train)

#%%
# Predict on the test set
y_pred_arstmade = model_arstmade.predict(X_test)

#%%
# Evaluate the model
accuracy_arstmade = accuracy_score(y_test, y_pred_arstmade)
print("Accuracy:", accuracy_arstmade)
print(classification_report(y_test, y_pred_arstmade))
# %%
