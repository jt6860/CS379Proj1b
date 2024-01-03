# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:44:09 2024
@author: John Torres
Course: CS379 - Machine Learning
Project 1b: Classifying The Titanic Dataset
Supervised Classification Algorithm: Random Forest Classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# Helper function
text_digit_vals = {}

def handle_non_numerical_data(df):
    columns = df.columns.values

    def convert_to_int(val):
        return text_digit_vals[val]

    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))

    return df


# Read the dataset
dataset = pd.read_excel('CS379T-Week-1-IP.xls', index_col=0)

# Drop unnecessary columns
dataset.drop(columns=['body', 'name', 'ticket', 'home.dest'], inplace=True)

# Handle non-numerical data
dataset = handle_non_numerical_data(dataset)

# Fill missing values with 0
dataset.fillna(0, inplace=True)

# Separate the predicting column from the whole dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode the predicting variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the data into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Check for presence of instances for each class in the predicting variable
unique_classes = np.unique(y)
missing_classes = []
for unique_class in unique_classes:
    if unique_class not in y_test:
        missing_classes.append(unique_class)

# Remove instances with missing classes from the test dataset
missing_indices = np.isin(y_test, missing_classes)
X_test = X_test[~missing_indices]
y_test = y_test[~missing_indices]

# Use the random forest classifier for the prediction
classifier = RandomForestClassifier()
classifier = classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

# Print the results
print('Confusion Matrix:')
print(confusion_matrix(y_test, predicted))
print('Accuracy Score:', accuracy_score(y_test, predicted))
print('Report:')
print(classification_report(y_test, predicted, zero_division=0))