import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def kernel(X1, X2, hyp):
    l = np.array([hyp[i] for i in range(X1.shape[1])])
    sigma_v = hyp[-1]
    dists = cdist(X1 / l, X2 / l, metric="sqeuclidean")
    K = sigma_v**2 * np.exp(-0.5*dists)
    return K

def cholesky(K):
    K = np.array(K, float)
    L = np.zeros_like(K)
    n = K.shape[0]
    for j in range(n):
        for i in range(j, n):
            if i == j:
                sumk = 0
                for k in range(j):
                    sumk += L[i,k]**2
                L[i,j] = np.sqrt(K[i,j]-sumk)
            else:
                sumk = 0
                for k in range(j):
                    sumk += L[i,k]*L[j,k]
                L[i,j] = 1/(L[j,j]) * (K[i,j] - sumk)

    return L

def log_likelihood(hyp):

    K = kernel(X_train, X_train, hyp) + np.eye(X_train.shape[0], X_train.shape[0]) * 1e-10
    L = cholesky(K)
    alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y_train))

    log_like = -0.5 * np.transpose(y_train) @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y_train) * np.log(2*np.pi)

    return -log_like

def my_func(X):
    f = (X[:,0] - 1)**2 + (X[:,1] - 1)**2 + (X[:,2] - 1)**2
    return f

def optimize_hyperparams(hyp):
    res = minimize(log_likelihood, hyp, method='L-BFGS-B', options = {'gtol': 1e-6, 'disp': False})
    hyp = res.x
    return hyp

def update_GP(X_train, X_test, hyp):

    K = kernel(X_train, X_train, hyp) + np.eye(X_train.shape[0], X_train.shape[0])*1e-10
    K_cross = kernel(X_train, X_test, hyp)
    K_test = kernel(X_test, X_test, hyp)

    L = cholesky(K)
    alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y_train))
    v = np.linalg.solve(L, K_cross)
    mu_post = np.transpose(K_cross) @ alpha
    s2_post = K_test - np.transpose(v) @ v

    mu_up = mu_post - np.diag(s2_post)
    mu_down = mu_post + np.diag(s2_post)

    return mu_post, s2_post

global X_train
global y_train

n = 11
n_dim = 3

a = -2
b = 2

# Grid for plotting
x = np.meshgrid(*[np.linspace(a, b, n) for dim in range(n_dim)])
X_test = np.transpose(np.vstack([x_i.ravel() for x_i in x]))

hyp = np.array([1, 1, 1, 1])

presamples = 15
X_train = np.random.uniform(low = a, high = b, size=(presamples,n_dim))
y_train = my_func(X_train)

iter_count = 0
err_list = []
beta = 1
err = 1

print(X_test)

while err > 1e-6:

    iter_count += 1
    print("Running iteration no: ", iter_count)
    # optimize hyperparameters
    hyp = optimize_hyperparams(hyp)

    # update gaussian process
    mu_post, s2_post = update_GP(X_train, X_test, hyp)

    a = mu_post - beta * np.diag(s2_post)

    X_new1 = X_test[np.argmin(a)]
    y_proj = np.min(a)

    X_train = np.append(X_train, X_new1.reshape(1, n_dim), axis=0)
    y_new = my_func(X_new1.reshape(1, n_dim))

    y_train = np.append(y_train, y_new, axis = 0)

    y_best = np.min(y_train)

    err = abs(y_best - y_proj)
    err_list.append(err)

    print("Error: ", err)

# # # Plotting
# Sci-kit reference

X_best = X_train[np.argmin(y_train)]
y_best = np.min(y_train)

print(X_best, y_best)

# kernel = 1 * RBF(length_scale = [1,1],length_scale_bounds=(1e-5, 1e2)) # (1e-3, 1e3)
# gaussian_process = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
# gaussian_process.fit(X_train, my_func(X_train))
# mu_pred, sigma_pred = gaussian_process.predict(X_test, return_std=True)

# fig, axs = plt.subplots(1,2, subplot_kw={'projection': '3d'}, dpi=300)
# # axs[0].plot(X_test[:,0], X_test[:,1], mu_pred, '.')
# axs[0].set_title('Scikit')
# axs[1].plot(X_test[:,0], X_test[:,1], mu_post, '.')
# axs[1].plot(X_train[:,0], X_train[:,1], y_train, 'r.')
# axs[1].set_title('Jompz')
# plt.show()















