# Car Price Predictor Using Linear Regression

This project implements a linear regression model to predict car prices based on mileage. It consists of two main parts:

1. **Training the model**: Using gradient descent to find the optimal parameters (`theta0` and `theta1`).
2. **Predicting car prices**: After training, the model can predict the price of a car given its mileage. The program also provides a visual representation of actual vs. predicted prices, including where a specific car fits in.

## Project Structure

- `data.csv`: The dataset containing mileage and car prices.
- `predict_price.py`: The main script that:
  - Trains a linear regression model using gradient descent.
  - Allows you to input the mileage of a car to predict its price.
  - Visualizes the actual prices and predicted prices on a graph.
  - Calculates the precision of the model using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **Root Mean Squared Error (RMSE)**.

## Requirements

To run this project, you need the following Python libraries:

- `numpy`: For numerical operations
- `pandas`: For reading and handling the dataset
- `matplotlib`: For visualizing the data and predictions
- `scikit-learn`: For calculating precision metrics

### Install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn

python predict_price.py

Example of toutput :

Enter the mileage of the car: 85000
Predicted price for 85000 km: 6200.34

Mean Absolute Error (MAE): 557.83
Mean Squared Error (MSE): 445645.32
Root Mean Squared Error (RMSE): 667.57
