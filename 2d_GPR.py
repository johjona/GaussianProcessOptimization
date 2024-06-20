import numpy as np
import matplotlib.pyplot as plt
import pyGPs
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from mpl_toolkits import mplot3d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern

def my_func(X):
    f = (X[:,0] - 1)**2 + (X[:,1] - 1)**2 + (X[:,2] - 1)**2
    return f

def cov(X1, X2, hyp):
    ndim = X1.shape[1]
    sigma_f = hyp[2]
    l = np.array([hyp[0], hyp[1]]) #, hyp[2]])
    k = np.zeros((X1.shape[0], X2.shape[0]))
    M = np.eye(ndim, ndim) * (1/l**2)

    for i, xi in enumerate(X1):
        for j, xj in enumerate(X2):
            r = xi - xj
            k[i, j] = sigma_f**2 * np.exp(-0.5 * r @ M @ r)

    return k

def control(X1, X2):

    model = pyGPs.GPR()
    k = pyGPs.cov.RBFard(D = X1.shape[1])
    model.setPrior(kernel = k)
    model.setOptimizer("Minimize", num_restarts=10, covRange=[(-5,1),(-5,1),(-5,1)], likRange=[(-10,1)])
    model.optimize(X1, y_train)
    model.predict(X2)

    hyp = [np.exp(item) for item in np.concatenate((model.covfunc.hyp, model.likfunc.hyp))]
    
    return hyp, model.ym, model.ys2

def determine_grid(up, low, ndim, n):

    x1 = np.linspace(low, up, n)
    x2 = np.linspace(low, up, n)
    x3 = np.linspace(low, up, n)
    x4 = np.linspace(low, up, n)

    print(x1)
    
    p = 0

    coords = np.zeros((n**ndim, ndim))

    for i, x_i in enumerate(x1):
        for j, x_j in enumerate(x2):
            for k, x_k in enumerate(x3):
                coords[p,:] = np.array([x_i, x_j, x_k])
                p += 1

                
    return coords

def determine_new_point(X_new1, inc):

    x = np.linspace(X_new1[0] - inc, X_new1[0] + inc, 21)
    y = np.linspace(X_new1[1] - inc, X_new1[1] + inc, 21)
    z = np.linspace(X_new1[2] - inc, X_new1[2] + inc, 21)

    X_temp = np.array([])
    k = 0

    for i, xi in enumerate(x):
        for j, xj in enumerate(y):
            for l, xl in enumerate(z):
                k += 1
                app = np.array([xi, xj, xl])
                if k == 1:
                    X_temp = np.hstack((X_temp, app))
                else:
                    X_temp = np.vstack((X_temp, app))


    mu_temp, sigma_temp = gaussian_process.predict(X_temp, return_std=True)

    alpha_temp = mu_temp - beta * sigma_temp

    tot_x = 0
    tot_y = 0
    tot_z = 0

    for i, value in enumerate(X_temp):
        tot_x += value[0] * alpha_temp[i]
        tot_y += value[1] * alpha_temp[i]
        tot_z += value[2] * alpha_temp[i]

    cog = np.array([tot_x/np.sum(alpha_temp), tot_y/np.sum(alpha_temp), tot_z/np.sum(alpha_temp)])

    mu_cog, sigma_temp = gaussian_process.predict(cog.reshape(1,-1), return_std=True)

    return cog, mu_cog, X_temp, mu_temp

global X_train, y_train 

n = 21
ndim = 3
a = -10
b = 10
presamples = 8

X_train = np.random.uniform(low = a, high = b, size=(presamples,ndim))
y_train = my_func(X_train)
y_best = []

inc = abs(a - b)/(n-1)

X_test = determine_grid(a, b, ndim, n)
err = 1

err_list = []

iter_count = 0

while err > 1e-4:

    iter_count += 1

    kernel = 1 * RBF(length_scale = 2*np.ones((ndim, 1)), length_scale_bounds=(1e-3, 100)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    gaussian_process.fit(X_train.reshape(-1,ndim), y_train)
    mu_3, sigma_pred = gaussian_process.predict(X_test, return_std=True)

    beta = 1
    alpha2 = mu_3 - beta * sigma_pred

    X_new1 = X_test[np.argmin(alpha2)]
    y_proj = np.min(alpha2)

    X_new1, y_proj, X_temp, mu_temp = determine_new_point(X_new1, inc)

    X_train = np.append(X_train, X_new1.reshape(1, ndim), axis=0)
    y_new = my_func(X_new1.reshape(1, ndim))
    y_train = np.append(y_train, y_new, axis = 0)

    y_best = np.min(y_train)

    err = abs(y_best - y_proj)

    err_list.append(err)

improvement = []

print(err_list)

print('Number of iterations including presampling: ', iter_count + presamples)
print('Final error: ', err_list[-1])
print('Best X-value: ', X_train[np.argmin(y_train)])

fig, ax = plt.subplots(1,1)
ax.plot(err_list, 'k.-')
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Absolute error')
plt.show()





