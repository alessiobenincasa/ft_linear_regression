# predict_prices.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

theta_data = pd.read_csv('model_parameters.csv')
theta0 = theta_data['theta0'].values[0]
theta1 = theta_data['theta1'].values[0]
X_mean = theta_data['X_mean'].values[0]
X_std = theta_data['X_std'].values[0]

data = pd.read_csv('data.csv')

X = data['km'].values
y = data['price'].values

X_norm = (X - X_mean) / X_std

def predict_price(mileage, theta0, theta1, X_mean, X_std):
    mileage_normalized = (mileage - X_mean) / X_std
    return theta0 + theta1 * mileage_normalized

mileage_input = float(input("Enter the mileage of the car: "))

predicted_price = predict_price(mileage_input, theta0, theta1, X_mean, X_std)

print(f"Predicted price for {mileage_input} km: {predicted_price}")

predicted_prices = theta0 + theta1 * X_norm

plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices, color='red', label='Predicted Prices')

plt.scatter(mileage_input, predicted_price, color='green', label=f"Your car ({mileage_input} km)", marker='x', s=100)

mae = mean_absolute_error(y, predicted_prices)
mse = mean_squared_error(y, predicted_prices)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Show plot
plt.title('Mileage vs Price')
plt.xlabel('Mileage (km)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
