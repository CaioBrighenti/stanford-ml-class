import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# read data from files
path = os.getcwd() + '\ex1\Python\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Area', 'Rooms', 'Cost'])
data = (data - data.mean()) / data.std()
data.head(10)
data.describe()

# prepare data for multivariate linear Regression
data.insert(0, 'Ones', 1)
m = data.shape[1]
# separate X and Y
X = data.iloc[:,0:m-1]
y = data.iloc[:,m-1:m]
# dataframe -> matrix
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))

# prepare for gradient descent
alpha = 0.01
num_iters = 1000

# normalize features
#X = featureNormalize(X)
# perform gradient descent
theta, cost = gradientDescent(X, y, theta, alpha, num_iters)

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    temp_theta = np.matrix(np.zeros(theta.shape))

    i = 1
    for i in range(num_iters):
        # error = hypothesis - actual dataset
        errors = (X * theta.T) - y
        # j = feature number
        for j in range(X.shape[1]):
            term = np.multiply(errors, X[:, j])
            temp_theta[0, j] = theta[0, j] - alpha / len(X) * np.sum(term)

        theta = temp_theta
        #J_history[i] = computeCost(X, y, theta)

    return theta, J_history

# normalize features before gradient descent
def featureNormalize(X):
    X_norm = X
    mu = X.mean(0)
    sigma = X.std(0)
    #X_norm = X_norm - mu
    #X_norm = X_norm / sigma
    for i in range(1, X.shape[1]):
        X_norm[:,i] = (X_norm[:,i] - mu[:,i])
        X_norm[:,i] = (X_norm[:,i] / sigma[:,i])
    return X_norm
