#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]

x = values[:, 7:]


# x = a1.normalize_data(x)



N_TRAIN = 100;
x_train = x[0:N_TRAIN, :]
print(x_train.shape)
print(targets.shape)
print(type(x_train))


x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


reg = LinearRegression()

reg.fit(x_train,t_train)

print(type(reg.coef_))

print(reg.coef_)
print(reg.coef_.shape)

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

# (w, tr_err) = a1.linear_regression()
# (t_est, te_err) = a1.evaluate_regression()


# Produce a plot of results.
# plt.plot(train_err.keys(), train_err.values())
# plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Test error', 'Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
