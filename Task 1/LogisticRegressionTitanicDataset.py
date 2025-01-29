import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset
data = pd.read_csv(r'D:\Shravani Study Files\RSM\Task1\Titanic_Dataset.csv')

# Check the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values in crucial columns
data.dropna(subset=['Age', 'Embarked'], inplace=True)

# Convert 'Sex' and 'Embarked' columns to numeric values
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Or use pd.get_dummies for one-hot encoding

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# X (features) and y (target)
X = data[features]
y = data[target]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the splits
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target values on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

# Plotting the confusion matrix
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

import joblib

# Save the trained model
joblib.dump(model, 'logistic_regression_model.pkl')

# Load the model and verify
loaded_model = joblib.load('logistic_regression_model.pkl')
print(f"Loaded Model Accuracy: {loaded_model.score(X_test, y_test)}")
