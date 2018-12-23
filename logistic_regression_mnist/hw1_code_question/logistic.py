""" Methods for doing logistic regression."""

import numpy as np
from .utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    y    = sigmoid(np.matrix(data) * np.matrix(weights))

    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function

    N    = np.shape(y)[0]
    ce   = -np.sum(np.transpose(targets) * np.log(y))

    y    = np.round(y)
    a    = np.isclose(targets, y)
    frac = a[a == True]

    frac_correct = np.shape(frac)[1] / float(N)

    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization']:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        """" TODO: compute f and df without regularization"""
        f = -np.transpose(targets) * np.log(y) - np.transpose(1 - targets) * np.log(1 - y)
        df = (np.transpose(data) * (y - targets))

    return f, df, y

def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    # [...]
    y = logistic_predict(weights, data)

    f = -np.transpose(targets) * np.log(y) - np.transpose(1 - targets) * np.log(1 - y)\
        + hyperparameters["weight_regularization"] * np.transpose(np.matrix(weights)) * np.matrix(weights)
    df = (np.transpose(data) * (y - targets)) + hyperparameters["weight_regularization"] * weights

    return f, df
