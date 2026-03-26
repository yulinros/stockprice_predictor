import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# 1. Download S&P 500 data
print("Downloading data...")
data = yf.download('^GSPC', start='2021-01-01', end='2026-01-01')

# Ensure we have a single index for columns (yfinance sometimes returns MultiIndex)
if isinstance(data.columns, pd.MultiIndex):
      data.columns = data.columns.get_level_values(0)

df = data[['Close']].copy()
# If Close is still a DataFrame (due to duplicate columns or MultiIndex residue), squeeze or select
if isinstance(df, pd.DataFrame) and df.shape[1] > 1:
      df = df.iloc[:, 0:1]

# 2. Feature Engineering
print("Engineering features...")

# Target: Daily Log Return (Stationary)
# Log return = ln(Price_t / Price_{t-1})
df['Log_Return_Target'] = np.log(df['Close'] / df['Close'].shift(1))

# Lag features (Log Returns) - 1 to 10 days
for i in range(1, 11):
      df[f'Lag_{i}'] = df['Log_Return_Target'].shift(i)

# Rolling Statistics (on log returns)
for window in [5, 10, 20]:
      df[f'Rolling_Mean_{window}'] = df['Log_Return_Target'].shift(1).rolling(window=window).mean()
      df[f'Rolling_Std_{window}'] = df['Log_Return_Target'].shift(1).rolling(window=window).std()

# Momentum (5-day and 10-day historical log returns)
df['Momentum_5'] = df['Log_Return_Target'].shift(1).rolling(window=5).sum()
df['Momentum_10'] = df['Log_Return_Target'].shift(1).rolling(window=10).sum()

# Volatility (20-day rolling std of returns) - already covered by Rolling_Std_20, but let's be explicit
df['Volatility_20'] = df['Log_Return_Target'].shift(1).rolling(window=20).std()

# Drop NaN values
df.dropna(inplace=True)

# Ensure columns are strings
df.columns = [str(col) for col in df.columns]

# Define Features and Target
feature_cols = [col for col in df.columns if 'Lag' in col or 'Rolling' in col or 'Momentum' in col or 'Volatility' in col]
print(f"Features developed: {len(feature_cols)}")
if len(feature_cols) == 0:
      print(f"DEBUG: df.columns = {df.columns.tolist()}")
      raise ValueError("No features were identified. Check column naming.")

X = df[feature_cols]
y = df['Log_Return_Target']

# 3. Data Splitting (STRICT)
X_train = X[X.index <= '2024-12-31']
y_train = y[y.index <= '2024-12-31']
X_test = X[(X.index >= '2025-01-01') & (X.index <= '2025-12-31')]
y_test = y[(y.index >= '2025-01-01') & (y.index <= '2025-12-31')]

# Capture indices and original prices for reconstruction
test_actual_prices = df.loc[X_test.index, 'Close']
# We need the price of the day BEFORE the test period to start reconstruction
# However, since we are predicting log returns for EACH day in 2025,
# price_t = price_{t-1} * exp(log_return_t)
# We can just use the actual price_{t-1} to see how the model does "one step ahead"
last_train_price = df[df.index <= '2024-12-31']['Close'].iloc[-1]
# For testing reconstruction:
test_prev_prices = data['Close'].shift(1).loc[X_test.index]

print(f"Training Period: {X_train.index.min().date()} to {X_train.index.max().date()}")
print(f"Testing Period:  {X_test.index.min().date()} to {X_test.index.max().date()}")

# 4. Preprocessing (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training & Hyperparameter Tuning
print("\nTuning Random Forest...")
rf_params = {
      'n_estimators': [50, 100],
      'max_depth': [10, 20]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='neg_mean_squared_error')
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_

print("Tuning XGBoost...")
xgb_params = {
      'n_estimators': [100, 200],
      'learning_rate': [0.01, 0.05],
      'max_depth': [3, 5]
}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=3, scoring='neg_mean_squared_error')
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_

# 6. Predictions (Log Returns)
rf_log_ret_preds = best_rf.predict(X_test_scaled)
xgb_log_ret_preds = best_xgb.predict(X_test_scaled)

# 7. Reconstruction (Convert Log Returns back to Prices)
# Price_pred = Price_yesterday * exp(Log_Return_pred)
rf_price_preds = test_prev_prices * np.exp(rf_log_ret_preds)
xgb_price_preds = test_prev_prices * np.exp(xgb_log_ret_preds)

# 8. Evaluation (MSE on Prices)
rf_mse = mean_squared_error(test_actual_prices, rf_price_preds)
xgb_mse = mean_squared_error(test_actual_prices, xgb_price_preds)

print(f"\nImproved Evaluation Results (MSE):")
print(f"Random Forest MSE: {rf_mse:.2f}")
print(f"XGBoost MSE:       {xgb_mse:.2f}")
print(f"Best RF Params:  {rf_grid.best_params_}")
print(f"Best XGB Params: {xgb_grid.best_params_}")

# Baseline comparison (from previous results)
print(f"\nBaseline MSE (Price Prediction):")
print(f"Old Random Forest: 227534.35")
print(f"Old XGBoost:       279709.53")

# 9. Visualization
plt.figure(figsize=(14, 7))
plt.plot(test_actual_prices.index, test_actual_prices.values, label='Actual Price', color='black', linewidth=2)
plt.plot(test_actual_prices.index, rf_price_preds, label='Improved Random Forest', linestyle='--', alpha=0.8)
plt.plot(test_actual_prices.index, xgb_price_preds, label='Improved XGBoost', linestyle=':', alpha=0.8)

plt.title('Improved S&P 500 Price Prediction - 2025')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('prediction_results_improved.png')
print("\nImproved plot saved as 'prediction_results_improved.png'")
