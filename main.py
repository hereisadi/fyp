import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from sklearn.metrics import mean_absolute_error

# Step 1: Download BTC-USD Data
df = yf.download("BTC-USD")

# Ensure the index is datetime and clean the data
df.index = pd.to_datetime(df.index)
df = df[['Adj Close']]

# Step 2: Train-Test Split
to_row = int(len(df) * 0.9)
training_data = list(df[0:to_row]['Adj Close'].values)
testing_data = list(df[to_row:]['Adj Close'].values)

# Step 3: ARIMA Modeling and Prediction
model_predictions = []
training_data_copy = training_data.copy()

for i in range(len(testing_data)):
    model = ARIMA(training_data_copy, order=(4, 1, 0))  # ARIMA order (p, d, q)
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)  # Forecast one step ahead
    yhat = output[0]  # Predicted value
    model_predictions.append(yhat)
    actual_test_value = testing_data[i]
    training_data_copy.append(actual_test_value)  # Append actual test value to training data

# Step 4: Visualization
plt.figure(figsize=(15, 9))
plt.grid(True)

date_range = df[to_row:].index
plt.plot(date_range, model_predictions, color='blue', marker="o", linestyle="dashed", label="BTC Predicted Price")
plt.plot(date_range, testing_data, color='red', label="BTC Actual Price")

plt.title("Bitcoin Price Prediction")
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.legend()
plt.show()

# mae = mean_absolute_error(testing_data, model_predictions)
# print(f"Mean Absolute Error (MAE): {mae}")
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(testing_data, model_predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
