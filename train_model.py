import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('data.csv')

# Feature and target
X = data['km'].values
y = data['price'].values

# Normalize the feature for better gradient descent performance
X_norm = (X - np.mean(X)) / np.std(X)

# Initialize parameters
theta0 = 0
theta1 = 0
learning_rate = 0.01
iterations = 1000

# Gradient Descent Function
def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = theta0 + theta1 * X
        error = predictions - y
        theta0 -= learning_rate * (1/m) * np.sum(error)
        theta1 -= learning_rate * (1/m) * np.sum(error * X)
    return theta0, theta1

# Find optimal thetas
theta0, theta1 = gradient_descent(X_norm, y, theta0, theta1, learning_rate, iterations)

print(f"Optimal Theta0: {theta0}")
print(f"Optimal Theta1: {theta1}")
