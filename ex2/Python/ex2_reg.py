# Some code adapted from Josh Wittenauer's Curious Insight blog
# describing his experience with the same exercises in Python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
%matplotlib inline

# Read data from file
path = os.getcwd() + '\ex2\Python\ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Microchip Test 1', 'Microchip Test 2', 'Passed'])
data.describe()

# plot data
positive = data[data['Passed'].isin([1])]
negative = data[data['Passed'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Microchip Test 1'], positive['Microchip Test 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Microchip Test 1'], negative['Microchip Test 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Microchip Test 1 Score')
ax.set_ylabel('Microchip Test 2 Score')

# map features
degree = 5
x1 = data['Microchip Test 1']
x2 = data['Microchip Test 2']
data.insert(3, 'Ones', 1)
for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
data.drop('Microchip Test 1', axis=1, inplace=True)
data.drop('Microchip Test 2', axis=1, inplace=True)
data.head()

#prepare matrices
cols = data.shape[1]
# Isolates X and Y
X = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]
# dataframes -> matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
[m, n] = X.shape
theta = np.zeros((n))

# set learning rate
learn_rate = 1

# use scipy library to minimize cost function
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=computeGradient, args=(X, y, learn_rate))
costFunction(result[0], X, y, learn_rate)


# vectorized sigmoid function implementation
def sigmoid(z):
    return np.power((np.exp(-z) +1), -1)

# compute gradient vector
def computeGradient(theta, X, y, learn_rate):
    theta = np.matrix(theta)
    hypot = sigmoid(X * theta.T)
    grad = np.zeros(theta.shape[1])
    for i in range(X.shape[1]):
        term = np.multiply((hypot - y), X[:, i])
        grad[i] = (1/m) * np.sum(term)
        if(i!=1):
            reg_term = (learn_rate / m) * theta[:,i]
            grad[i] = grad[i] + reg_term
    return grad

# computes cost
def costFunction(theta, X, y, learn_rate):
    theta = np.matrix(theta)
    m = y.shape[0]
    hypot = sigmoid(X * theta.T)
    errors = np.multiply(-y, np.log(hypot)) - np.multiply((1 - y),np.log(1 - hypot))
    reg_term = (learn_rate / (2 * m)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    J = 1 / m * sum(errors) + reg_term
    return J

def predict(theta, X):
    theta = np.matrix(theta)
    return sigmoid(X * theta.T)
