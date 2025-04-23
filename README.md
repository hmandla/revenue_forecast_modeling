import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# 1. Load your revenue data
# Sample DataFrame creation (you can replace this with your CSV)
data = {
    'Date': pd.date_range(start='2020-01-01', periods=36, freq='M'),
    'Revenue': [100 + i*2 + (i%6)*5 for i in range(36)]  # Simulated pattern
}
df = pd.DataFrame(data).set_index('Date')

# 2. Split into train and test
train = df[:-6]
test = df[-6:]

# 3. Fit ARIMA model (simple order for demo)
model = ARIMA(train['Revenue'], order=(2, 1, 2))
model_fit = model.fit()

# 4. Forecast next 6 months
forecast = model_fit.forecast(steps=6)

# 5. Evaluate performance
mae = mean_absolute_error(test['Revenue'], forecast)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 6. Plot actual vs forecast
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['Revenue'], label='Actual')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.title("Revenue Forecast using ARIMA")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.show()
