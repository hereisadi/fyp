# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Excel file
file_path = "data_from_sir.xlsx"  # Replace with your file path
data = pd.ExcelFile(file_path)

# Load the 'tp' (total precipitation) sheet
tp_data = data.parse('tp')

# Convert 'time' column to datetime and aggregate data (mean precipitation per month)
tp_data['time'] = pd.to_datetime(tp_data['time'])
tp_time_series = tp_data.groupby(tp_data['time'].dt.to_period('M'))['tp'].mean()
tp_time_series.index = tp_time_series.index.to_timestamp()

# Train-test split: 90% train, 10% test
to_row = int(len(tp_time_series) * 0.9)
train_data = tp_time_series.iloc[:to_row]
test_data = tp_time_series.iloc[to_row:]

# Optimize ARIMA parameters using auto_arima
auto_model = auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
print("Optimal ARIMA Order:", auto_model.order)

# Fit ARIMA model using the optimal parameters
model_predictions = []
train_data_list = list(train_data)

for i in range(len(test_data)):
    model = ARIMA(train_data_list, order=auto_model.order)
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    yhat = output[0]
    model_predictions.append(yhat)
    train_data_list.append(test_data.iloc[i])

# Evaluate Model
mae = mean_absolute_error(test_data, model_predictions)
mse = mean_squared_error(test_data, model_predictions)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot Actual vs. Predicted values
plt.figure(figsize=(15, 9))
plt.grid(True)

# Plot actual values
plt.plot(test_data.index, test_data, color='red', label="Actual Total Precipitation")

# Plot predicted values
plt.plot(test_data.index, model_predictions, color='blue', linestyle="dashed", label="Predicted Total Precipitation")

plt.title("Total Precipitation Forecasting with ARIMA")
plt.xlabel("Date")
plt.ylabel("Total Precipitation")
plt.legend()
plt.show()


# Import libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from pmdarima import auto_arima

# # Load the Excel file
# file_path = "data_from_sir.xlsx"  # Replace with your file path
# data = pd.ExcelFile(file_path)

# # Load the 'tp' (total precipitation) sheet
# tp_data = data.parse('tp')

# # Convert 'time' column to datetime and aggregate data (mean precipitation per month)
# tp_data['time'] = pd.to_datetime(tp_data['time'])
# tp_time_series = tp_data.groupby(tp_data['time'].dt.to_period('M'))['tp'].mean()
# tp_time_series.index = tp_time_series.index.to_timestamp()

# # Train-test split: 90% train, 10% test
# to_row = int(len(tp_time_series) * 0.9)
# train_data = tp_time_series.iloc[:to_row]
# test_data = tp_time_series.iloc[to_row:]

# # Optimize ARIMA parameters using auto_arima
# auto_model = auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
# print("Optimal ARIMA Order:", auto_model.order)

# # Fit ARIMA model to the full dataset
# final_model = ARIMA(tp_time_series, order=auto_model.order)
# final_model_fit = final_model.fit()

# # Forecast for the next 10 years (120 months if monthly data)
# future_periods = 120  # 10 years * 12 months
# forecast = final_model_fit.forecast(steps=future_periods)

# # Create a DataFrame for the forecasted values
# forecast_index = pd.date_range(start=tp_time_series.index[-1] + pd.DateOffset(months=1), 
#                                periods=future_periods, freq='M')
# forecast_df = pd.DataFrame({'Forecasted Precipitation': forecast}, index=forecast_index)

# # Plot the historical data and the forecasted values
# plt.figure(figsize=(15, 9))
# plt.grid(True)

# # Plot historical data
# plt.plot(tp_time_series, label="Historical Data", color='blue')

# # Plot forecast
# plt.plot(forecast_df, label="10-Year Forecast", color='orange', linestyle="dashed")

# plt.title("10-Year Rainfall Forecast")
# plt.xlabel("Date")
# plt.ylabel("Total Precipitation")
# plt.legend()
# plt.show()

# # Display forecasted values
# print("Forecasted Precipitation for the Next 10 Years:")
# print(forecast_df)
