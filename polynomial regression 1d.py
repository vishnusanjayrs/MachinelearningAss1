#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]

x = values[:, 7:]

N_TRAIN = 100;

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

train_err = []
test_err = []
feat=8
for i in range(0, 8):
    x_train = x[0:N_TRAIN, i]
    x_test = x[N_TRAIN:, i]
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', degree=3)
    train_err.append((feat, tr_err))
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree=3)
    test_err.append((feat, te_err))
    feat+=1

train_err = np.array(train_err)
test_err = np.array(test_err)
print(train_err)
print(test_err)
# Produce a plot of results.
plt.bar(train_err[:, 0], train_err[:, 1])
plt.bar(test_err[:, 0], test_err[:, 1])
plt.ylabel('RMS')
plt.legend(['Training error', 'Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('features')
plt.show()
