# -*- coding: utf-8 -*-
"""
Created on Tue Jan 2 15:44:09 2024
@author: John Torres
Course: CS379 - Machine Learning
Project 1b: Classifying The Titanic Dataset
Supervised Classification Algorithm: Random Forest Classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

def preprocess_data():
    # Read the Excel document and convert it to a DataFrame for preprocessing. Reset index to move 'pclass' to its own column.
    df = pd.read_excel('CS379T-Week-1-IP.xls', index_col=0)
    df = df.reset_index(names=['pclass'])

    # Drop unused columns.
    drop_columns = ['body', 'name', 'ticket', 'home.dest', 'cabin']
    df.drop(columns=drop_columns, inplace=True)

    # Fill NaN values with specific values, X's for 'boat' and 'embarked', 0s for remaining.
    new_values = {"boat": "X", "embarked": "X"}
    df.fillna(value=new_values, inplace=True)
    df.fillna(0, inplace=True)

    # Encoding for non-numerical columns, ensure uniform str types.
    non_numerical_columns = ['sex', 'embarked', 'boat']
    encoder = LabelEncoder()
    for column in non_numerical_columns:
        df[column] = encoder.fit_transform(df[column].astype('str'))

    return df

def remove_missing_classes(X_test, y_test, missing_classes):
    # Remove instances with missing classes from the test dataset.
    missing_indices = np.isin(y_test, missing_classes)
    X_test = X_test[~missing_indices]
    y_test = y_test[~missing_indices]

    return X_test, y_test

def train_random_forest(X_train, y_train):
    # Instantiate and fit the Random Forest Classifier for prediction.
    classifier = RandomForestClassifier()
    classifier = classifier.fit(X_train, y_train)

    return classifier

def evaluate_model(classifier, X_test, y_test):
    # Make predictions using the trained classifier.
    predicted = classifier.predict(X_test)
    confusionMatrix = confusion_matrix(y_test, predicted)
    classes = classifier.classes_
    
    # Check if there are missing classes in the test dataset.
    missing_classes = set(classes) - set(y_test)
    if missing_classes:
        # Remove instances with missing classes from the test dataset.
        X_test, y_test = remove_missing_classes(X_test, y_test, missing_classes)
        # Update the classes accordingly.
        classes = np.unique(y_test)
    
    # Print Accuracy Score and Classification Report, plot Confusion Matrix heatmap.
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, predicted))
    print('Accuracy Score:', accuracy_score(y_test, predicted))
    print('Report:')
    print(classification_report(y_test, predicted, zero_division=0))
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot(include_values=True, xticks_rotation='vertical')
    
    plt.show()

## Main workflow    
# Preprocess the data.
df = preprocess_data()

# Separate the predicting column ('survived') from the remainder of the dataset.
X = df.drop('survived', axis=1)
y = df['survived']

# Encode the predicting dataset.
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the data into test and train datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Check for presence of instances with missing classes for each class in the predicting set.
unique_classes = np.unique(y)
missing_classes = []
for unique_class in unique_classes:
    if unique_class not in y_test:
        missing_classes.append(unique_class)

# Remove any instances with missing classes from the test dataset.
X_test, y_test = remove_missing_classes(X_test, y_test, missing_classes)

# Train the random forest classifier on the preprocessed data.
classifier = train_random_forest(X_train, y_train)

# Evaluate the model.
evaluate_model(classifier, X_test, y_test)