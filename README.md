# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
### Date: 02-09-2025
### Developed By: A JEEVITH
### Register Number: 212223240059
### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:
```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv(r"avocado.csv",nrows=35)
data=df['Total Volume'].values

N=len(data)
lags=range(35)

#Pre-allocate autocorrelation table
autocorr_values=[]
#Mean
mean_data=np.mean(data)
#Variance
variance_data=np.var(data)
#Normalized data
normalized_data=(data-mean_data)/np.sqrt(variance_data)
#Go through lag components one-by-one
for lag in lags:
    if lag == 0:
        autocorr_values.append(1)
    else:
        auto_cov = np.sum((data[:-lag] - mean_data) * (data[lag:] - mean_data)) / N
        autocorr_values.append(auto_cov / variance_data)
#display the graph
plt.figure(figsize=(10, 6))
plt.stem(lags, autocorr_values)
plt.title('Autocorrelation of Data')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()
```
### OUTPUT:
<img width="956" height="545" alt="image" src="https://github.com/user-attachments/assets/9c0735f1-cadb-42ef-933e-ed379ac81c8e" />

### RESULT:
Thus we have successfully implemented the auto correlation function in python.
