# Some code adapted from Josh Wittenauer's Curious Insight blog
# describing his experience with the same exercises in Python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Read data from file
path = os.getcwd() + '\ex1\Python\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head(10)
data.describe()
data.plot(kind="scatter", x='Population', y='Profit', figsize=(12,8))

# Prepare data for linear regression
data.insert(0, 'Ones', 1)
cols = data.shape[1]
# Isolates X and Y
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols-1:cols]
# dataframes -> matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

#computeCost(X, y, theta)

# prepare variables for gradient descen
alpha = 0.01
num_iters = 1000

# perform gradient descent
g, cost = gradientDescent(X, y, theta, alpha, num_iters)

#visualize data
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(num_iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

# function to compute J
def computeCost(X, y, theta):
    m = len(X)
    predict = X * theta.T
    errors = np.power((predict - y), 2)
    return np.sum(errors) / (2*m)

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    temp_theta = np.matrix(np.zeros(theta.shape))

    for i in range(num_iters):
        errors = (X * theta.T) - y

        for j in range(X.shape[1]):
            term = np.multiply(errors, X[:, j])
            temp_theta[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp_theta
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
