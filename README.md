# revenue_forecast_modeling
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load sample data
df = pd.read_csv("revenue_trend.csv")  # Columns: Year, Revenue
X = df[['Year']]
y = df['Revenue']

# Model
model = LinearRegression()
model.fit(X, y)
future_years = pd.DataFrame({'Year': np.arange(2025, 2031)})
predicted_revenue = model.predict(future_years)

# Plot
plt.scatter(X, y, color='blue', label="Actual Revenue")
plt.plot(future_years, predicted_revenue, color='red', label="Forecast")
plt.title("Revenue Forecast")
plt.xlabel("Year")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.show() 
