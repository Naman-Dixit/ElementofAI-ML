# -*- coding: utf-8 -*-
"""machine learning b50exp09.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11sxvHFatSrli3xnzBkYA38kgD9SsbSEn
"""

import pandas as pd
import io
import numpy as np
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

uploaded = files.upload()

file_name = list(uploaded.keys())[0]
data = pd.read_csv(io.BytesIO(uploaded[file_name]))

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Check class distribution
print("Class distribution:\n", y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a basic Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
print("Baseline Model without Handling Imbalance")
print(classification_report(y_test, y_pred))

# Logistic Regression with class weights
lr_weighted = LogisticRegression(class_weight='balanced')
lr_weighted.fit(X_train, y_train)

# Evaluate
y_pred_weighted = lr_weighted.predict(X_test)
print("Model with Class Weights")
print(classification_report(y_test, y_pred_weighted))

from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train model on SMOTE-resampled data
lr_smote = LogisticRegression()
lr_smote.fit(X_train_smote, y_train_smote)

# Evaluate
y_pred_smote = lr_smote.predict(X_test)
print("Model with SMOTE Oversampling")
print(classification_report(y_test, y_pred_smote))

from imblearn.under_sampling import RandomUnderSampler

# Apply Random Undersampling
undersample = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

# Train model on undersampled data
lr_under = LogisticRegression()
lr_under.fit(X_train_under, y_train_under)

# Evaluate
y_pred_under = lr_under.predict(X_test)
print("Model with Random Undersampling")
print(classification_report(y_test, y_pred_under))

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Calculate AUC-ROC
y_prob = lr_smote.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("AUC-ROC Score with SMOTE Oversampling:", roc_auc)
