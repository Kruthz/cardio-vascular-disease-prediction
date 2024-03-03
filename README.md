# Cardiovascular Disease Prediction Project

## Overview
This project aims to develop a machine learning model to predict the risk of cardiovascular diseases (CVDs) based on various health-related features. Cardiovascular diseases are a leading cause of mortality globally, emphasizing the importance of early detection and risk assessment. By leveraging machine learning techniques, this project seeks to assist healthcare professionals in identifying individuals at higher risk of developing CVDs.

## Dataset
The dataset used is Farmingham CVD dataset can be downloaded at https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset

## Features
- **Data Preprocessing**: The dataset is loaded and preprocessed to handle missing values, outliers, and standardize features. Irrelevant attributes are removed, and rows with missing values are dropped.
- **Exploratory Data Analysis (EDA)**: Visualizations are used to explore the distribution of data and relationships between features and the target variable.
- **Model Training and Evaluation**: Several classifiers including Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, Logistic Regression, Random Forest, and Support Vector Machine are trained and evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Hyperparameter tuning is performed for some models using GridSearchCV.
- **Feature Selection**: Feature selection is performed using SelectFromModel with Random Forest and SelectKBest with chi-square test. The impact of feature selection on model performance is analyzed.
- **Handling Class Imbalance**: Synthetic Minority Over-sampling Technique (SMOTE) is used to handle class imbalance.
- **Results Comparison**: The performance of models with and without feature selection using different methods is compared.

## Usage
1. **Data Preprocessing**: Load the dataset and preprocess it using the provided script. Ensure missing values are handled appropriately, and irrelevant attributes are removed.
2. **Exploratory Data Analysis (EDA)**: Explore the dataset using visualizations to understand the distribution of data and relationships between features and the target variable.
3. **Model Training and Evaluation**: Train various classifiers on the dataset and evaluate their performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
4. **Feature Selection**: Experiment with different feature selection methods such as SelectFromModel and SelectKBest to identify the most relevant features for predicting CVD risk.
5. **Handling Class Imbalance**: If the dataset exhibits class imbalance, consider using techniques such as SMOTE to balance the classes.
6. **Results Comparison**: Compare the performance of models with and without feature selection to determine the most effective approach for predicting CVD risk.

## Future Work
- **Model Optimization**: Continuously optimize the model by experimenting with different algorithms, hyperparameters, and feature selection techniques to improve prediction accuracy.
- **Integration**: Integrate the trained model into healthcare systems for real-time risk assessment and decision support.
- **Validation**: Validate the model's performance using external datasets and collaborate with healthcare professionals for real-world validation.
- **Extension**: Extend the scope of the model to incorporate additional features and risk factors for a more comprehensive assessment of cardiovascular disease risk.
