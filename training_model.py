import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')

X = data['km'].values
y = data['price'].values

X_norm = (X - np.mean(X)) / np.std(X)

theta0 = 0
theta1 = 0
learning_rate = 0.01
iterations = 1000

def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = theta0 + theta1 * X
        error = predictions - y
        theta0 -= learning_rate * (1/m) * np.sum(error)
        theta1 -= learning_rate * (1/m) * np.sum(error * X)
    return theta0, theta1

theta0, theta1 = gradient_descent(X_norm, y, theta0, theta1, learning_rate, iterations)

theta_data = pd.DataFrame({
    'theta0': [theta0],
    'theta1': [theta1],
    'X_mean': [np.mean(X)],
    'X_std': [np.std(X)]
})
theta_data.to_csv('model_parameters.csv', index=False)

print(f"Training complete. Optimal Theta0: {theta0}, Theta1: {theta1}")
