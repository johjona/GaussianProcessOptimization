 #%% 

import numpy as np
import matplotlib.pyplot as plt
import pyGPs
from mpl_toolkits import mplot3d
from scipy.stats import multivariate_normal
from timeit import default_timer as timer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern

def dist(X1, X2):
    diff = (X1.reshape(-1, 1, 2) - X2)
    r = np.linalg.norm(diff, ord=2, axis=-1)
    return 

def SE(sigma_G, r, l):
    s2 = sigma_G*np.exp(-0.5*r**2/l**2)
    return s2

# def Matern(sigma_G, r, l):
#     s2 = (1 + np.sqrt(3)*r/l)*np.exp(-np.sqrt(3)*r/l)  
#     return s2

def my_func(X):
    x1 = X[:,0]
    x2 = X[:,1]
    f = x1**2 + x2**2 + 1
    return f

def RQ(X1, X2):
    alpha = 1
    diff = (X1.reshape(-1, 1, 2) - X2)/np.array([2,2])
    r = np.linalg.norm(diff, ord=2, axis=-1)
    s2 = (1 + r**2/(2*alpha))**(-alpha)
    return s2

def gamma_exp(X1, X2):
    gamma = 2
    diff = (X1.reshape(-1, 1, 2) - X2)/np.array([2,2])
    r = np.linalg.norm(diff, ord=2, axis=-1)
    s2 = np.exp(-r**gamma)
    return s2

def plot_kernels(n, l, X_pred):

    x1, x2 = np.meshgrid(np.linspace(-2,2,n), np.linspace(-2,2,n))
                        
    X1 = np.transpose(np.vstack([x1.ravel(), x2.ravel()]))

    l = 1
    sigma_G = 1

    s2 = np.zeros((X1.shape[0], 1))
    for i in range(X1.shape[0]):
        xi = X1[i]
        ri = dist(xi, np.array([0,0]))
        s2[i,0] = SE(sigma_G, ri, l)

    s2 = np.array(s2)
    
    fig, ax = plt.subplots(1,1, subplot_kw={'projection': '3d'}, dpi=200)
    ax.plot_surface(x1, x2, s2.reshape(n,n))
    ax.set_axis_off()

    s2 = np.zeros((X1.shape[0], 1))
    for i in range(X1.shape[0]):
        xi = X1[i]
        ri = dist(xi, np.array([0,0]))
        s2[i,0] = Matern(ri, l)

    s2 = np.array(s2)

    fig, ax = plt.subplots(1,1, subplot_kw={'projection': '3d'}, dpi=200)
    ax.plot_surface(x1, x2, s2.reshape(n,n))
    ax.set_axis_off()

def plot_true_obj(axs, n):
    x1, x2 = np.meshgrid(np.linspace(-2,2,n), np.linspace(-2,2,n))
    f = x1**2 + x2**2
    axs[2].plot_surface(x1, x2, f.reshape(n, n))

def preprocessing(n):

    x1, x2 = np.meshgrid(np.linspace(-2,2,n), np.linspace(-2,2,n))
    X_pred = np.transpose(np.vstack([x1.ravel(), x2.ravel()]))

    # plot_kernels(n, 2, X_pred)

    X_training = np.array([[1,1],
                        [-1.5, 1],
                        [0.3, -0.3]])

    # plot_true_obj(X_pred, n)

    return X_pred, x1, x2

def mean_prediction(X_pred, X_obs, y_obs, sigma_V, sigma_G, l):
    
    r_obs = dist(X_obs, X_obs)
    r_cross = dist(X_obs, X_pred)
    r_pred = dist(X_pred, X_pred)

    sigma_obs = SE(sigma_G, r_obs, l) + np.eye(X_obs.shape[0])*sigma_V
    sigma_cross = SE(sigma_G, r_cross, l)
    sigma_pred = SE(sigma_G, r_pred, l)
    
    mu_pred = np.transpose(sigma_cross) @ np.linalg.inv(sigma_obs) @ y_obs 
    sigma_pred = sigma_pred - np.transpose(sigma_cross) @ np.linalg.inv(sigma_obs) @ sigma_cross
    sigma_pred = np.sqrt(np.diagonal(sigma_pred))
    return mu_pred, sigma_pred

def optimize_hyper(X_train, y_train):
    
    model = pyGPs.GPR()
    model.setOptimizer("Minimize", num_restarts=20)
    k = pyGPs.cov.RBFard(D = 2)
    model.optimize(X_train, y_train)

    l1 = np.exp(model.covfunc.hyp)[0]

    print(k.hyp)
    
    sigma_G = np.exp(model.covfunc.hyp)[1]
    sigma_V = np.exp(model.likfunc.hyp)

    print(l1, sigma_G, sigma_V)
    
    return l1, sigma_G, sigma_V

def GPoptimization(X_train, y_train, n):

    i = 0
    err = 1

    # while err > 1e-4:
    for i in range(5):

        i += 1
        X_pred, x1, x2 = preprocessing(n)

        # model = pyGPs.GPR()
        # k = pyGPs.cov.Matern()
        # model.setPrior(kernel = k)
        # model.setOptimizer('Minimize', num_restarts=10)
        # model.optimize(X_train, y_train)
        # model.predict(X_pred)

        # # print(model.covfunc.hyp)

        # mu_pred = model.ym
        # sigma_pred = model.ys2

        # l, sigma_G, sigma_V = optimize_hyper(X_train, y_train)
        # mu_pred, sigma_pred = mean_prediction(X_pred, X_train, y_train, sigma_V, sigma_G, l)

        kernel = 2*Matern(length_scale = [1.0], length_scale_bounds=(1e-3, 1e3))
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gaussian_process.(X_train.reshape(-1,2), y_train)
        mu_pred, sigma_pred = gaussian_process.predict(X_pred, return_std=True)

        beta = 1
        alpha = mu_pred - beta*sigma_pred

        X_new = X_pred[np.argmin(alpha), None]

        X_train = np.append(X_train, X_new, axis = 0)
        y_train = np.append(y_train, my_func(X_new), axis = 0)

        y_best = np.min(y_train)
        y_proj = np.min(alpha)

        err = abs((y_best - y_proj)/y_best)
# 
        print(err, i)

    return mu_pred, sigma_pred, x1, x2, alpha, X_train, y_train

n = 11
X_train = np.array([[1,2],
                    [-1.5,0.3],
                    [0.6, 1.5]])

y_train = my_func(X_train)

mu_pred, sigma_pred, x1, x2, alpha, X_train, y_train = GPoptimization(X_train, y_train, n)

fig, axs = plt.subplots(1,3, subplot_kw={'projection': '3d'}, dpi=200)
axs[0].plot_surface(x1, x2, mu_pred.reshape(n, n), alpha = 0.25, lw = 0.5, edgecolor = 'royalblue')
axs[0].plot_surface(x1, x2, mu_pred.reshape(n, n) + sigma_pred.reshape(n,n), alpha = 0.25, lw = 0.5, edgecolor = 'royalblue')
axs[0].plot_surface(x1, x2, mu_pred.reshape(n, n) - sigma_pred.reshape(n,n), alpha = 0.25, lw = 0.5, edgecolor = 'royalblue')
axs[0].scatter(X_train[:-1,0], X_train[:-1,1], y_train[0:-1], color='red')

axs[1].plot_surface(x1, x2, alpha.reshape(n,n), alpha = 0.25, lw = 0.5, edgecolor = 'royalblue')

print(X_train[np.argmin(y_train)])

plot_true_obj(axs, n)

plt.show()



