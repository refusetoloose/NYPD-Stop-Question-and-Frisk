# New York City Stop, Question, and Frisk (SQF) Data Analysis Project

## Overview
This project focuses on analyzing the NYPD Stop, Question, and Frisk (SQF) data to understand the patterns and factors associated with police encounters. The dataset contains information about individuals stopped by the police, including demographic details, reasons for the stop, outcomes, and whether force was used. Various machine learning models are applied to predict different aspects of police encounters, such as the likelihood of arrest and the potential use of force.

## Dataset
The dataset used in this project is sourced from the NYPD Stop, Question, and Frisk Database. It includes records of police encounters in New York City, providing details such as demographic information (age, sex, race), circumstances of the stop, reasons for the stop, outcomes (e.g., arrest, summons), and whether force was used.

## Files
1. `2012_EDA.py`: This Python script performs exploratory data analysis (EDA) on the 2012 dataset, including data cleaning, visualization, and statistical analysis.
2. `Pred_armed.py`: Python script that builds and evaluates a machine learning model to predict whether an individual was armed during a police encounter.
3. `pred_arrests.py`: Python script for predicting the likelihood of arrest during a police encounter using machine learning techniques.
4. `pred_force.py`: Python script to predict the potential use of force during a police encounter based on various factors using a Decision Tree Classifier.
5. `README.md`: This file contains an overview of the project, details about the dataset, and descriptions of the files in the repository.
6. `df_bin.csv`: Cleaned dataset with binary-encoded features.
7. `df_clean.csv`: Cleaned dataset with categorical and numerical features.
8. `README.md` (Completed): The README file for the GitHub repository, including project overview, dataset details, file descriptions, and machine learning models used.

## Reports Overview:
1. **Exploratory Data Analysis (EDA):**
   - Detailed analysis of the dataset to understand its structure, distributions, and key characteristics.
   - Findings regarding demographic patterns, encounter outcomes, and spatial-temporal insights.

2. **Association Rule Mining:**
   - Utilization of association rule mining techniques to discover patterns and relationships in police encounters.
   - Insights into demographic attributes' relationships and their implications for law enforcement strategies.

3. **Cluster Analysis:**
   - Identification of spatial clusters in crime locations and grouping of stopped individuals based on encounter reasons.
   - Exploration of potential applications and insights derived from clustering techniques.

4. **Predictive Modeling:**
   - Application of machine learning algorithms for predicting armed encounters and likelihood of arrests.
   - Evaluation of models' performance using metrics such as accuracy, precision, recall, and F1-score.

## Machine Learning Models
- **Armed Prediction Model**: Logistic Regression model to predict whether an individual was armed during a police encounter based on frisking, searching, and other factors.
- **Arrest Prediction Model**: Logistic Regression model to predict the likelihood of arrest during a police encounter based on various factors, including frisking, searching, and contraband.
- **Force Use Prediction Model**: Decision Tree Classifier to predict the potential use of force during a police encounter based on features such as frisking, searching, and previous force actions.

## Model Performance:
- **Predicting Armed Encounters:**
  - Random Forest Classifier: Accuracy ≈ 92%, Precision ≈ 8%, Recall ≈ 53%, F1-score ≈ 14%.
- **Predicting Arrests:**
  - Logistic Regression: Accuracy ≈ 96%, Precision ≈ 80%, Recall ≈ 40%, F1-score ≈ 53%.
- **Decision Tree Classifier:** Accuracy ≈ 90%.

## Conclusion:
This project provides valuable insights into New York City's SQF data, including demographic patterns, spatial-temporal trends, and predictive modeling for law enforcement purposes. The findings can inform policy decisions, resource allocation, and interventions aimed at improving policing strategies and promoting community safety.





