import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
%matplotlib inline

# Read data from file
path = os.getcwd() + '\ex2\Python\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.describe()

# prepare matrices
data.insert(0, 'Ones', 1)
cols = data.shape[1]
# Isolates X and Y
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols-1:cols]
# dataframes -> matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
[m, n] = X.shape
theta = np.zeros((n))

# use scipy library to minimize cost function
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=computeGradient, args=(X, y))
costFunction(result[0], X, y)


# predict new value given minimized theta
# Expected value: 0.775 +/- 0.002
test = np.matrix('1 45 85')
predict(result[0], test)

# vectorized sigmoid function implementation
def sigmoid(z):
    return np.power((np.exp(-z) +1), -1)

# compute gradient vector
def computeGradient(theta, X, y):
    theta = np.matrix(theta)
    hypot = sigmoid(X * theta.T)
    grad = np.zeros(theta.shape[1])
    for i in range(X.shape[1]):
        term = np.multiply((hypot - y), X[:, i])
        grad[i] = (1/m) * np.sum(term)
    return grad

# computes cost
def costFunction(theta, X, y):
    theta = np.matrix(theta)
    m = y.shape[0]
    hypot = sigmoid(X * theta.T)
    errors = np.multiply(-y, np.log(hypot)) - np.multiply((1 - y),np.log(1 - hypot))
    J = 1 / m * sum(errors)
    return J

def predict(theta, X):
    theta = np.matrix(theta)
    return sigmoid(X * theta.T)
