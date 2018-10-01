#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import math

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]

x = values[:, 7:]


x = a1.normalize_data(x)


N_TRAIN = 100;
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
train_err =[]
test_err =[]
for p in range(1,7):
    print(p)
    (w, tr_err) = a1.linear_regression(x_train,t_train,'polynomial',degree=p)
    train_err.append((p,tr_err))
    (t_est, te_err) = a1.evaluate_regression(x_test,t_test,w,'polynomial',degree=p)
    test_err.append((p,te_err))

train_err =np.array(train_err)
test_err =np.array(test_err)
# Produce a plot of results.
plt.plot(train_err[:,0], train_err[:,1])
plt.plot(test_err[:,0], test_err[:,1])
plt.ylabel('RMS')
plt.legend(['Test error', 'Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
