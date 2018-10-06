#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]

x = values[:, 7:]

N_TRAIN = 100

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

x_train = x[0:N_TRAIN, 3]
x_test = x[N_TRAIN:, 3]

print(x_train)

i_basis ='ReLU'
i_degree = 0

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

train_err = []
test_err = []

(w, tr_err) = a1.linear_regression(x_train, t_train, i_basis, degree=i_degree)
train_err.append((1, tr_err))

(t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, i_basis, degree=i_degree)
test_err.append((1, te_err))

train_err = np.array(train_err)
test_err = np.array(test_err)
print(train_err)
print(test_err)
# Produce a plot of results.
plt.plot(train_err[:, 0], train_err[:, 1], 'bo')
plt.plot(test_err[:, 0], test_err[:, 1], 'gx')
plt.ylabel('RMS')
plt.legend(['Training error', 'Test error'])
plt.title('Relu training and testing error')
plt.xlabel('features')
plt.show()

t_dummy = np.linspace(1, 500, 500)
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500).reshape(-1, 1)
# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
y_ev, _ = a1.evaluate_regression(x_ev, t_dummy, w, 'ReLU', degree=0)
plt.plot(x_ev, y_ev, '-r')
plt.plot(x_train, t_train, 'bo')
plt.plot(x_test, t_test, 'gx')
plt.title('A visualization of a regression estimate using random outputs for feature' + features[10])
plt.show()

