# ML Metrics Demo

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

y_true = [10, 20, 30, 40, 50]
y_predicted = [12, 18, 33, 37, 48]

# Mean Absolute Error Calculation
mae = mean_absolute_error(y_true, y_predicted)

# Mean Squared Error Calculation
mse = mean_squared_error(y_true, y_predicted)

# Root Mean Squared Error Calculation
rmse = mse ** 0.5

# R-squared Calculation
r2 = r2_score(y_true, y_predicted)

print("Mean Absolute Error, MAE :",mae)
print("Mean Squared Error, MSE :",mse)
print("Root Mean Squared Error, RMSE :",rmse)
print("R-squared :",r2)