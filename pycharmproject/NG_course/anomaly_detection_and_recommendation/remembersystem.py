import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(context='notebook', style='white', palette=sns.color_palette("RdBu"))
import scipy.io as sio

""" Notes: X - num_movies (1682)  x num_features (10) matrix of movie features  
    Theta - num_users (943)  x num_features (10) matrix of user features  
    Y - num_movies x num_users matrix of user ratings of movies  
    R - num_movies x num_users matrix, where R(i, j) = 1 if the  
    i-th movie was rated by the j-th user  """

movies_mat = sio.loadmat("./data/ex8_movies.mat")
Y, R = movies_mat.get('Y'), movies_mat.get("R")
m, u = Y.shape
n = 10

param_mat = sio.loadmat("./data/ex8_movieParams.mat")
theta, X = param_mat.get("Theta"), param_mat.get('X')


def serialize(X, theta):
    return np.concatenate((X.ravel(), theta.ravel()))


def deserialize(param, n_movie, n_user, n_features):
    return param[:n_movie * n_features].reshape(n_movie, n_features), \
           param[n_movie * n_features:].reshape(n_user, n_features)


def cost(param, Y, R, n_features):
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner, 2).sum() / 2


users = 4
movies = 5
features = 3

X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

param_sub = serialize(X_sub, theta_sub)
cost(param_sub, Y_sub, R_sub, features)

param = serialize(X, theta)

cost(param, Y, R, n)

n_movie, n_user = Y.shape


def gradient(param, Y, R, n_features):
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    X_grad = inner @ theta

    theta_grad = inner.T @ X

    return serialize(X_grad, theta_grad)


X_grad, theta_grad = deserialize(gradient(param, Y, R, 10), n_movie, n_user, 10)


def regularized_cost(param, Y, R, n_features, l=1.):
    reg_term = np.power(param, 2).sum() * (l / 2)

    return cost(param, Y, R, n_features) + reg_term


regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5)
regularized_cost(param, Y, R, 10, l=1)


def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param
    return grad + reg_term


X_grad, theta_grad = deserialize(regularized_gradient(param, Y, R, 10), n_movie, n_user, 10)

movie_list = []
with open("./data/movie_ids.txt", encoding="latin-1") as f:
    for line in f:
        tokens = line.strip(" ")
        movie_list.append(" ".join(tokens[1:]))
movie_list = np.array(movie_list)

ratings = np.zeros((1682))
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

Y, R = movies_mat.get('Y'), movies_mat.get('R')

Y = np.insert(Y, 0, ratings, axis=1)  # now I become user 0
print(Y.shape)

R = np.insert(R, 0, ratings != 0, axis=1)
print(R.shape)

n_features = 50
n_movie, n_user = Y.shape
l = 10

X = np.random.standard_normal((n_movie, n_features))
theta = np.random.standard_normal((n_user, n_features))
print(X.shape)
print(theta.shape)

param = serialize(X, theta)

# Y_norm = Y - Y.mean()
Y_norm = Y - Y.mean(axis=1).reshape(Y.shape[0], 1)

import scipy.optimize as opt

res = opt.minimize(fun=regularized_cost,
                   x0=param,
                   args=(Y_norm, R, n_features, l),
                   method='TNC',
                   jac=regularized_gradient)

print("res:", res)

x_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)
print(x_trained.shape)
print(theta_trained.shape)

prediction = x_trained @ theta_trained.T

my_preds = prediction[:, 0] + Y.mean()
my_preds = prediction[:, 0] + Y.mean(axis=1)

idx = np.argsort(my_preds)[::-1]

# top ten idx
print(my_preds[idx[:10]])

for m in movie_list[idx][:10]:
    print(m)
