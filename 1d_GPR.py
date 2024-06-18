import numpy as np
import matplotlib.pyplot as plt
import pyGPs
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def my_func(X):
    f = X * np.sin(X)
    return f

def cov(X1, X2, hyp):
    ndim = X1.shape[1]
    sigma_f = hyp[1]
    sigma_n = hyp[2]
    l = hyp[0]*np.ones((ndim, 1))
    k = np.zeros((X1.shape[0], X2.shape[0]))
    M = np.eye(ndim, ndim) * (1/l**2)

    for i, xi in enumerate(X1):
        for j, xj in enumerate(X2):
            r = xi - xj
            k[i, j] = sigma_f**2 * np.exp(-0.5 * r @ M @ r)

    # k = k + np.eye(X1.shape[0], X1.shape[0]) * sigma_n**2

    return k

def control(X1, X2):

    model = pyGPs.GPR()
    k = pyGPs.cov.RBF()
    model.setPrior(kernel = k)
    model.setOptimizer("Minimize", num_restarts=20)
    model.optimize(X1, y_train)
    model.predict(X2)

    hyp = [np.exp(item) for item in np.concatenate((model.covfunc.hyp, model.likfunc.hyp))]

    print(hyp)

    return hyp, model.ym, model.ys2

global X_train, y_train

sigma_n = 1

n_dim = 1
n_pred = 100

X_train = np.random.uniform(low = 0, high = 10, size=(3,1))
X_test = np.linspace(0, 10, n_pred).reshape(n_pred, n_dim)

y_train = my_func(X_train)

hyp, mu_pred, sigma_pred = control(X_train, X_test)
# sigma_pred = np.sqrt(sigma_pred)

k_train = cov(X_train, X_train, hyp) + np.eye(X_train.shape[0], X_train.shape[0]) * hyp[2]**2
k_cross = cov(X_train, X_test, hyp)
k_test = cov(X_test, X_test, hyp)

mu_post = np.transpose(k_cross) @ np.linalg.inv(k_train) @ y_train
sigma_post = np.sqrt(np.diag(k_test - np.transpose(k_cross) @ np.linalg.inv(k_train) @ k_cross))

beta = 1
alpha_post = mu_post - beta * sigma_post.reshape(100,1)

alpha_pred = mu_pred - beta * np.sqrt(sigma_pred)

                     
fig, ax = plt.subplots(2,2)
ax[0, 0].plot(X_test, my_func(X_test), 'k--', alpha = 0.3)
ax[0, 0].plot(X_train, y_train, 'b.')
ax[0, 0].plot(X_test, mu_post, 'b-', alpha = 0.3)
low = mu_post.reshape(100,) - sigma_post.reshape(100,)
high = mu_post.reshape(100,) + sigma_post.reshape(100,)
ax[0, 0].fill_between(X_test.reshape(100,), low.reshape(100,), high.reshape(100,), alpha = 0.3)
ax[1, 0].plot(X_test, alpha_post, 'r-')

print(X_test.shape, alpha_post.shape)

ax[0, 1].plot(X_test, my_func(X_test), 'k--', alpha = 0.3)
ax[0, 1].plot(X_test, mu_pred, 'b-', alpha = 0.3)
ax[1, 1].plot(X_test, alpha_pred, 'r-')
low = mu_pred - np.sqrt(sigma_pred)
high = mu_pred + np.sqrt(sigma_pred)
ax[0, 1].fill_between(X_test.reshape(100,), low.reshape(100,), high.reshape(100,), alpha = 0.3)

# ax[0].legend(['True function', 'Training points', 'My predicted', 'PyGPs predicted'])
# ax[1].legend(['True function', 'Training points', 'My predicted', 'PyGPs predicted'])

ax[0, 0].set_ylim(-10, 10)
ax[0, 1].set_ylim(-10, 10)
plt.show()






