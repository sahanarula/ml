#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 18:47:22 2018

@author: sahilnarula
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
y_sc = StandardScaler()
X = X_sc.fit_transform(X)
y = y_sc.fit_transform(y.reshape(len(y), 1))

# SVR
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, y)

# Predict Data
predicted_value = y_sc.inverse_transform(regressor.predict(X_sc.transform(6.5)))

# Plot the data
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.xlabel("Positions")
plt.ylabel("Salaries")
plt.title("Positions vs Salaries (SVR)")
plt.show()