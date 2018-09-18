#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:10:10 2018

@author: sahilnarula
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Fitting Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 4)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Visualize linear regression output
import matplotlib.pyplot as plt
plt.scatter(X, y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.xlabel("Salaries")
plt.ylabel("Positions")
plt.title("Position vs Salaries (Linear Regression)")
plt.show()


# Visualize polynomial regression output
plt.scatter(X, y, color = "red")
plt.plot(X, poly_reg.predict(poly_features.fit_transform(X)), color = "blue")
plt.xlabel("Salaries")
plt.ylabel("Positions")
plt.title("Position vs Salaries (Polynomial Regression)")
plt.show()

#Advanced grid of Xs in the entire range of input
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, poly_reg.predict(poly_features.fit_transform(X_grid)), color = "blue")
plt.xlabel("Salaries")
plt.ylabel("Positions")
plt.title("Position vs Salaries (Polynomial Regression)")
plt.show()

# Predicting using Linear Regression
regressor.predict(6.5)

# Predicting using Polynomial Regression
poly_reg.predict(poly_features.fit_transform(6.5))