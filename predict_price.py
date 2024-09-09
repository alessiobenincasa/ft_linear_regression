import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def predict_price(mileage, theta0, theta1, X_mean, X_std):
    mileage_normalized = (mileage - X_mean) / X_std
    return theta0 + theta1 * mileage_normalized

mileage_input = float(input("Enter the mileage of the car: "))

X_mean = np.mean(X)
X_std = np.std(X)

predicted_price = predict_price(mileage_input, theta0, theta1, X_mean, X_std)

print(f"Predicted price for {mileage_input} km: {predicted_price}")

X_norm = (X - np.mean(X)) / np.std(X)

predicted_prices = theta0 + theta1 * X_norm

# Plot the dataset
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices, color='red', label='Predicted Prices')

plt.scatter(mileage_input, predicted_price, color='green', label=f"Your car ({mileage_input} km)", marker='x', s=100)

mae = mean_absolute_error(y, predicted_prices)
print(f"Mean Absolute Error (MAE): {mae}")

mse = mean_squared_error(y, predicted_prices)
print(f"Mean Squared Error (MSE): {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

plt.title('Mileage vs Price')
plt.xlabel('Mileage (km)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
