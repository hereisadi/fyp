import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your dataset (update 'file_path' to the correct location on your system)
file_path = 'data_from_sir.xlsx'
sheet_name = 'tp'  # Use the relevant sheet name

# Load the data and preprocess
data = pd.read_excel(file_path, sheet_name=sheet_name)
data['time'] = pd.to_datetime(data['time'])
data = data.groupby('time')['tp'].mean().reset_index()  # Aggregating rainfall data by timestamp
data.set_index('time', inplace=True)

# Resample to monthly data
monthly_tp = data.resample('M').mean()

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(monthly_tp) * 0.8)
train, test = monthly_tp.iloc[:train_size], monthly_tp.iloc[train_size:]

# Define and fit the SARIMA model
seasonal_order = (1, 1, 1, 12)  # Seasonal component: (P, D, Q, s)
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order)
sarima_fitted = sarima_model.fit(disp=False)

# Forecast on the testing set
sarima_forecast = sarima_fitted.forecast(steps=len(test))

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
mae = mean_absolute_error(test, sarima_forecast)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(test.values.flatten(), sarima_forecast.values)
r2 = r2_score(test, sarima_forecast)

# Print evaluation metrics
print(f'SARIMA Model Evaluation Metrics:')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape:.2f}%')
print(f'R-squared: {r2}')

# Residual Analysis
residuals = test.values.flatten() - sarima_forecast.values

# Plot histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Plot ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=20)
plt.title('ACF of Residuals')
plt.grid()
plt.show()

# Train-Test RMSE Comparison
train_forecast = sarima_fitted.fittedvalues
train_rmse = np.sqrt(mean_squared_error(train, train_forecast))
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {rmse}')

# Confidence Interval Analysis for the test set
sarima_future_forecast = sarima_fitted.get_forecast(steps=len(test))
future_conf_int = sarima_future_forecast.conf_int()
lower_bounds = future_conf_int.iloc[:, 0]
upper_bounds = future_conf_int.iloc[:, 1]
within_bounds = ((test.values.flatten() >= lower_bounds) & 
                 (test.values.flatten() <= upper_bounds)).mean()
print(f'Percentage of actual values within confidence intervals: {within_bounds * 100:.2f}%')

# Forecast for the next 10 years (120 months)
future_steps = 120
sarima_future_forecast = sarima_fitted.get_forecast(steps=future_steps)
future_forecast_values = sarima_future_forecast.predicted_mean
future_conf_int = sarima_future_forecast.conf_int()

# Plot the forecast for the next 10 years
plt.figure(figsize=(12, 6))
plt.plot(monthly_tp, label='Observed Data', color='blue')
plt.plot(future_forecast_values, label='Future Forecast', color='green')
plt.fill_between(future_conf_int.index,
                 future_conf_int.iloc[:, 0],
                 future_conf_int.iloc[:, 1], color='green', alpha=0.2, label='Confidence Interval')
plt.title('SARIMA Model Forecast for Next 10 Years')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.grid()
plt.show()

# Save the forecast for future reference
future_forecast_values.to_csv('sarima_future_forecast.csv', header=True)
print("Future forecast saved as 'sarima_future_forecast.csv'")
