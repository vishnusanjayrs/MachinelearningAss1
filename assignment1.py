"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats
import math


def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='cp1252')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:, 0]
    values = data.values[:, 1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values, dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)

    return (x - mvec) / stdvec


def linear_regression(x, t, basis, reg_lambda=0, degree=0):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    # e.g. phi = design_matrix(x,basis, degree)

    # TO DO:: Compute coefficients using phi matrix

    phi = design_matrix(x, basis, degree)

    print(phi.shape)
    print(type(phi))

    w_len = x.shape[1] * degree + 1

    w = np.ones(shape=(1, w_len), dtype=np.float32)

    w = np.linalg.pinv(phi).dot(t)

    # Measure root mean squared error on training data.

    y = phi.dot(w)

    mse_error = np.mean(np.square(t - y))

    train_err = mse_error

    return (w, train_err)


def design_matrix(x, basis, degree=0):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    # TO DO:: Compute desing matrix for each of the basis functions

    if basis == 'polynomial':
        for p in range(0, degree + 1):
            if p == 0:
                phi = np.ones(shape=x.shape[0])
                phi.shape = (-1, 1)
                print(phi.shape)
            else:
                temp = np.power(x, p)
                phi = np.concatenate((phi, temp), 1)
    elif basis == 'ReLU':
        phi = None
    else:
        assert (False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x, t, w, basis, degree):
    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset 
      """
    # TO DO:: Compute t_est and err

    phi = design_matrix(x,'polynomial',degree)

    t_est = phi.dot(w)

    err = np.mean(np.square(t - t_est))


    return (t_est, err)
