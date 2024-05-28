#%%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
#%%
# Load the dataset
df = pd.read_csv("df_bin.csv")
#%%
# Define features (X) and target variable (y)
X = df[['frisked', 'searched', 'contrabn', 'arstmade', 'pf_hands', 'pf_wall', 
        'pf_grnd', 'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_pepsp', 'pf_other']]
y = df['forceuse']

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Encode the categorical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

#%%
# Build the decision tree classifier model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train_encoded)

#%%
# Predict on the test set
y_pred = model.predict(X_test)

#%%
# Evaluate the model (you can use classification metrics here)
# For example, accuracy
accuracy = (y_pred == label_encoder.transform(y_test)).mean()
print("Accuracy:", accuracy)
# %%