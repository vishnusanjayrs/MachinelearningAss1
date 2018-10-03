#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]

x = values[:, 7:]

input_lambda = [0, .01, .1, 1, 10, 100, 1000, 10000]

x = a1.normalize_data(x)

n_train_valid = 100  # no of train and validation data sets
n_train = 90  # no of train data sets
n_start_valid = 0  # array starts at 0
n_end_valid = 10
n_valid = 10  # no of validation data sets
k = 10  # k-fold cross validation
i_degree = 2
i_basis = 'polynomial'

x_train_valid = x[0:n_train_valid, :]
x_test = x[n_train_valid:, :]
t_train_valid = targets[0:n_train_valid]
t_test = targets[n_train_valid:]

validation_error=[]
lambda_val_error=[]

min_valid_err =0
for idx in range(0,len(input_lambda)):
    for k_idx in range(0, k):
        x_valid = x_train_valid[n_start_valid:n_end_valid, :]
        t_valid = t_train_valid[n_start_valid:n_end_valid, :]
        if n_start_valid == 0:
            x_train = x_train_valid[n_end_valid:, :]
            t_train = t_train_valid[n_end_valid:, :]
        elif n_start_valid == n_train:
            x_train = x_train_valid[0:n_start_valid, :]
            t_train = t_train_valid[0:n_start_valid, :]
        else:
            x_train1 = x_train_valid[0:n_start_valid, :]
            x_train2 = x_train_valid[n_end_valid:, :]
            x_train = np.concatenate((x_train1, x_train2), 0)
            t_train1 = t_train_valid[0:n_start_valid, :]
            t_train2 = t_train_valid[n_end_valid:, :]
            t_train = np.concatenate((t_train1, t_train2), 0)

        (w, tr_err) = a1.linear_regression(x_train, t_train, i_basis,input_lambda[idx], i_degree)
        (t_est, te_err) = a1.evaluate_regression(x_valid, t_valid, w, i_basis,i_degree)
        validation_error.append(te_err)

    avg_valid_error = np.mean(validation_error)
    lambda_val_error.append((input_lambda[idx],avg_valid_error))
    if idx ==0:
        min_valid_err = avg_valid_error
        best_lambda = input_lambda[idx]
    else:
        if avg_valid_error <min_valid_err:
            min_valid_err=avg_valid_error
            best_lambda=input_lambda[idx]

print(lambda_val_error)
print(best_lambda)

lambda_val_error =np.array(lambda_val_error)
# Produce a plot of results.
plt.plot(lambda_val_error[:,0], lambda_val_error[:,1])
plt.ylabel('RMS')
plt.legend('Validation error')
plt.title('Validation error with different lambdas')
plt.xlabel('Regularizer co-eff lambda')
plt.show()

