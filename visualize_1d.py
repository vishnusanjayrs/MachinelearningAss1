#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
# Select a single feature.

t_train = targets[0:N_TRAIN]

print(features)



t_test = targets[N_TRAIN:]

t_dummy = np.linspace(1,500,500)

for i in range(3,7):
    x_train = x[0:N_TRAIN,i]
    x_test = x[N_TRAIN:,i]
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', degree=3)
    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500).reshape(-1,1)
    # TO DO:: Put your regression estimate here in place of x_ev.
    # Evaluate regression on the linspace samples.
    y_ev, _  = a1.evaluate_regression(x_ev, t_dummy, w, 'polynomial', degree=3)
    plt.plot(x_ev,y_ev,'-r')
    plt.plot(x_train,t_train,'bo')
    plt.plot(x_test,t_test,'.g')
    plt.title('A visualization of a regression estimate using random outputs for feature'+features[i+7])
    plt.show()
