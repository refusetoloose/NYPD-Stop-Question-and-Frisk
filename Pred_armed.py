# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
data_raw = pd.read_csv('df_clean.csv')
data_raw

# %%
# Check the content and missing values for the identified weapon-related columns
selected_weapon_columns = ['pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap']
weapon_info = data_raw[selected_weapon_columns].describe(include='all')

# Display the summary of weapon-related columns
weapon_info

# %%
# Create a binary target variable 'armed'
# Assign 1 if any weapon-related column has 'Y', otherwise 0
data_raw['armed'] = ((data_raw['pistol'] == 'Y') | (data_raw['riflshot'] == 'Y') | (data_raw['asltweap'] == 'Y') | (data_raw['knifcuti'] == 'Y') | (data_raw['machgun'] == 'Y') | (data_raw['othrweap'] == 'Y')).astype(int)

# Check the distribution of the new 'armed' column
armed_distribution = data_raw['armed'].value_counts()

# Display the distribution of the 'armed' target variable
armed_distribution

# %%
# Feature selection based on relevance and initial analysis
selected_features = ['trhsloc', 'perobs', 'frisked', 'searched', 'contrabn', 'inout', 'sex', 'race', 'height', 'build']

# Extract the selected features and target variable
feature_data = data_raw[selected_features + ['armed']]

# Check for missing values in the selected features
missing_values = feature_data.isnull().sum()

# Display the missing values information
missing_values

# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical, numerical, binary columns
categorical_columns = ['trhsloc', 'inout', 'sex', 'race', 'build']
numerical_columns = ['perobs', 'height'] 
binary_columns = ['frisked', 'searched', 'contrabn']

for col in binary_columns:
    data_raw[col] = (data_raw[col] == 'Y').astype(int)

# %%
# Create transformers for categorical and numerical data
categorical_transformer = OneHotEncoder(drop='first')  # Use drop='first' to avoid dummy variable trap
numerical_transformer = StandardScaler()

# Combine transformers into a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Apply transformations to the selected features
X = feature_data.drop('armed', axis=1)  # Features
y = feature_data['armed']  # Target variable

# Create a preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_transformed = pipeline.fit_transform(X)

# Display the shape of the transformed feature matrix to confirm the changes
X_transformed.shape

# %%
X_transformed


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define the model: Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Define the cross-validation strategy: StratifiedKFold to maintain the proportion of the target class
cv_strategy = StratifiedKFold(n_splits=5)

# Define scoring metrics
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
}

# Perform cross-validation using multiple scoring metrics
cv_results = cross_validate(rf_classifier, X_transformed, y, cv=cv_strategy, scoring=scoring_metrics)

# Collect average scores across all cross-validation folds for each metric
average_scores = {metric: scores.mean() for metric, scores in cv_results.items() if 'test_' in metric}

# Display the average scores
average_scores

# %%
# Re-select features for correlation analysis including the target 'armed'
selected_features_for_correlation = ['trhsloc', 'perobs', 'frisked', 'searched', 'contrabn', 'inout', 'sex', 'race', 'height', 'build', 'armed']

# Extract these features from the dataset
correlation_data = data_raw[selected_features_for_correlation]

# Explicitly convert binary and categorical variables for correlation analysis
categorical_to_convert = ['frisked', 'searched', 'contrabn', 'sex', 'race', 'inout', 'trhsloc', 'build']

# Convert 'Y'/'N' binary features to 0/1 and other categorical features to numerical codes
for col in categorical_to_convert:
    # If the feature is known to be binary and stored as 'Y'/'N'
    if data_raw[col].dtype == object and sorted(data_raw[col].unique()) == ['N', 'Y']:
        data_raw[col] = (data_raw[col] == 'Y').astype(int)
    else:
        # Convert other categorical features to category codes
        data_raw[col] = data_raw[col].astype('category').cat.codes

# Now recalculate the correlation matrix with these adjustments
correlation_matrix = data_raw[categorical_to_convert + ['armed']].corr()

# Display the updated correlation matrix
correlation_matrix

# %%
# Check value counts for each binary and categorical variable
value_counts = {col: data_raw[col].value_counts() for col in categorical_to_convert}
value_counts

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare the features (X) and target (y)
X = data_raw[selected_features]  # Assuming 'selected_features' includes all the features you want to use
y = data_raw['armed']            # The target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Extract feature importance
feature_importances = rf.feature_importances_
features = X_train.columns
importance_dict = dict(zip(features, feature_importances))

# Display sorted features by their importance
sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
sorted_importance

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Make predictions on the test data
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Display the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

# %%
