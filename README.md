# Ex.No: 03 COMPUTE THE AUTO FUNCTION(ACF)
## Date: 09.09.2025

## Name: A JEEVITH
## Register Number: 212223240059

### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.

### ALGORITHM:
1. Import the necessary packages (pandas, numpy, matplotlib, sklearn).
2. Load the dataset and convert the 'Formatted Date' column to datetime format.
3. Create a continuous time variable YearFrac as year + day_of_year / 365.
4. Drop rows with missing Temperature or YearFrac values.
5. Compute the mean and variance of the temperature data.
6. For each lag (0 to 34), compute autocorrelation:
7. Lag 0 → autocorrelation = 1.
8. Lag > 0 → compute normalized covariance of lagged data.
9. Store autocorrelation values in an array.
10. Plot the autocorrelation values against lag using a stem plot.

### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("weatherHistory.csv")
df.head()

# Convert date safely
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True, errors='coerce')

# Create YearFrac column
df['YearFrac'] = df['Formatted Date'].dt.year + (df['Formatted Date'].dt.dayofyear / 365)

# Drop rows with missing Temperature or YearFrac
df = df.dropna(subset=['Temperature (C)', 'YearFrac'])

# Define X and y
X_vals = df['YearFrac'].values.reshape(-1, 1)
data = df['Temperature (C)'].values

# ---- Linear Regression ----
lin_model = LinearRegression()
lin_model.fit(X_vals, data)
linear_pred = lin_model.predict(X_vals)

# ---- Polynomial Regression (Degree 2) ----
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X_vals)
poly2_model = LinearRegression()
poly2_model.fit(X_poly2, data)
poly2_pred = poly2_model.predict(X_poly2)

# ---- Autocorrelation ----
N = len(data)
lags = range(35)
autocorr_values = []
mean_data = np.mean(data)
variance_data = np.var(data)

for lag in lags:
    if lag == 0:
        autocorr_values.append(1)
    else:
        auto_cov = np.sum((data[:-lag] - mean_data) * (data[lag:] - mean_data)) / N
        autocorr_values.append(auto_cov / variance_data)

plt.figure(figsize=(10, 6))
plt.stem(lags, autocorr_values, use_line_collection=True)
plt.title("Autocorrelation of Temperature Data")
plt.xlabel("Lag (years)")
plt.ylabel("Autocorrelation")
plt.grid(True)
plt.show()
```

### OUTPUT:

### Dataset:

![alt text](<Images/image copy 2.png>)

### Graph:

![alt text](<Images/image copy.png>)

### Table of Autocorrelation Values:

![alt text](Images/image.png)

### RESULT:

Thus we have successfully implemented the auto correlation function in python.
