import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools
import warnings
warnings.filterwarnings("ignore")

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

# Define the parameter ranges for SARIMA
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
P = range(0, 3)
D = range(0, 2)
Q = range(0, 3)
s = [12]  # Monthly seasonality

# Generate all parameter combinations
parameters = list(itertools.product(p, d, q, P, D, Q, s))

# Function to evaluate SARIMA models
def evaluate_sarima(train, test, parameters):
    best_rmse = float("inf")
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    for param in parameters:
        try:
            # Extract parameters
            order = (param[0], param[1], param[2])
            seasonal_order = (param[3], param[4], param[5], param[6])
            
            # Fit SARIMA model
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            # Forecast on the test set
            forecast = fitted_model.forecast(steps=len(test))
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test, forecast))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
                best_seasonal_order = seasonal_order
                best_model = fitted_model
        except:
            continue
    return best_order, best_seasonal_order, best_rmse, best_model

# Optimize SARIMA hyperparameters
best_order, best_seasonal_order, best_rmse, best_model = evaluate_sarima(
    train, test, parameters
)

# Print the best parameters and RMSE
print(f"Best SARIMA Order: {best_order}")
print(f"Best Seasonal Order: {best_seasonal_order}")
print(f"Best RMSE: {best_rmse}")

# forecasting for next 10 years
future_steps = 120  # Forecast for 10 years (120 months)
future_forecast = best_model.get_forecast(steps=future_steps)
future_forecast_values = future_forecast.predicted_mean
future_conf_int = future_forecast.conf_int()

# plotting using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(monthly_tp, label='Observed Data', color='blue')
plt.plot(future_forecast_values, label='Future Forecast', color='green')
plt.fill_between(future_conf_int.index,
                 future_conf_int.iloc[:, 0],
                 future_conf_int.iloc[:, 1], color='green', alpha=0.2, label='Confidence Interval')
plt.title('Optimized SARIMA Model Forecast for Next 10 Years')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.grid()
plt.show()

# Evaluate model performance on training and testing sets
train_forecast = best_model.fittedvalues
train_rmse = np.sqrt(mean_squared_error(train, train_forecast))
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {best_rmse}")

# Saving the forect to a CSV file
future_forecast_values.to_csv('optimized_sarima_forecast.csv', header=True)
print("Future forecast saved as 'optimized_sarima_forecast.csv'")
