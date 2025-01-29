# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('D:/Shravani Study Files/RSM/Task1/USA_Housing_Dataset.csv')

# Check the first few rows
print(data.head())

# Select features and target
features = ['sqft_living']
target = 'price'

# X (features) and y (target)
X = data[features]
y = data[target]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the splits
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Check the model's coefficients and intercept
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

# Predict the target values on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R^2 (Coefficient of determination)
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

# Plotting Actual vs Predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)

# Plotting the line of perfect prediction (where actual = predicted)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

# Adding labels and title
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)

# Show the plot
plt.show()

# Save the model to a .pkl file
joblib.dump(model, 'linear_regression_model.pkl')

# Verifying the saved model by loading it again
loaded_model = joblib.load('linear_regression_model.pkl')

# Check the loaded model's coefficients and intercept
print(f"Intercept of loaded model: {loaded_model.intercept_}")
print(f"Coefficient of loaded model: {loaded_model.coef_}")
