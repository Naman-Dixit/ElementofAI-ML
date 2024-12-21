import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

import io
from google.colab import files # Import the files module from google.colab


# Upload the CSV file using the file upload widget
uploaded = files.upload()

# Get the filename of the uploaded file
filename = list(uploaded.keys())[0]
# Load the data into a Pandas DataFrame called 'df'
df = pd.read_csv(io.BytesIO(uploaded[filename])) # Assuming it's a CSV file. If it's Excel, use pd.read_excel instead.


# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Statistical Summary
print(df.describe())


# Handle missing values using SimpleImputer (replace missing with median)
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Detect and visualize outliers using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_imputed)
plt.title('Outlier Detection using Boxplots')
plt.show()


# Feature scaling using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_imputed.drop('Outcome', axis=1))

# One-hot encoding for categorical variables (if any)
# In this case, no categorical variables in PIMA dataset, but here's how you would do it for a categorical column:
# encoder = OneHotEncoder()
# encoded_features = encoder.fit_transform(df_imputed[['categorical_column']])

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
pca_features = pca.fit_transform(scaled_features)

# Visualize the PCA components
plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df_imputed['Outcome'], cmap='coolwarm')
plt.title('PCA of PIMA Diabetes Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

# Separate features and target variable
X = df_imputed.drop('Outcome', axis=1)
y = df_imputed['Outcome']

# Apply SMOTE for oversampling the minority class
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the class distribution after SMOTE
print(pd.Series(y_resampled).value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Linear Regression): {mse}')

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_class = rf_model.predict(X_test)

# Evaluate the model (Accuracy, Classification Report)
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Accuracy (Random Forest): {accuracy}')
print(f'Classification Report (Random Forest):\n{classification_report(y_test, y_pred_class)}')

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Distribution of Outcome (Diabetes Yes/No)')
plt.show()

# Visualize the correlation matrix
corr_matrix = df_imputed.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Save the imputed and preprocessed dataset to a CSV
df_imputed.to_csv('pima_imputed.csv', index=False)

