import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_periods = 24 # 2 years of monthly data
dates = pd.date_range(start='2023-01-01', periods=num_periods, freq='M')
sales = 1000 + 200 * np.sin(2 * np.pi * np.arange(num_periods) / 12) + 50 * np.random.randn(num_periods) # Seasonal trend with noise
df = pd.DataFrame({'Date': dates, 'Sales': sales})
# Add additional features (example: promotional events)
promotions = np.random.choice([0, 1], size=num_periods, p=[0.8, 0.2]) # 20% chance of promotion
df['Promotion'] = promotions
# --- 2. Data Cleaning and Feature Engineering ---
# In a real-world scenario, this section would involve handling missing values, outliers, etc.
# For this synthetic data, no cleaning is needed.
# Create lagged features for time series analysis
df['Sales_Lag1'] = df['Sales'].shift(1)
df['Sales_Lag12'] = df['Sales'].shift(12) # Lagged sales from the same month last year
df = df.dropna() # Drop rows with NaN values due to lagging
# --- 3. Predictive Modeling ---
# Simple linear regression model (can be replaced with more sophisticated models like ARIMA, Prophet etc.)
X = df[['Sales_Lag1', 'Sales_Lag12', 'Promotion']]
y = df['Sales']
X = sm.add_constant(X) # Add a constant term
model = sm.OLS(y, X).fit()
print(model.summary())
# --- 4. Visualization ---
# Plot actual vs. predicted sales
y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], y, label='Actual Sales')
plt.plot(df['Date'], y_pred, label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename = 'actual_vs_predicted.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 5. Model Evaluation (example: RMSE) ---
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse}")
# --- 6.  Further Analysis (Example: Seasonal Decomposition) ---
decomposition = sm.tsa.seasonal_decompose(df['Sales'], model='additive')
fig = decomposition.plot()
plt.tight_layout()
output_filename = 'seasonal_decomposition.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# Note:  This is a simplified example.  A real-world project would involve more rigorous model selection,
# hyperparameter tuning, and a more comprehensive evaluation.  Feature engineering would also be significantly
# more extensive based on the available data.