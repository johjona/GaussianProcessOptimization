import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import multivariate_normal
from timeit import default_timer as timer

def my_func(X):
    x = X[:,0]
    y = X[:,1]
    f = x**2 + y**2 + 1
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

def SE2(X1, X2):
    # broadcasting method
    diff = (X1.reshape(-1, 1, 2) - X2)/np.array([2,2])
    r = np.linalg.norm(diff, ord=2, axis=-1)
    s2 = np.exp(-0.5*r**2)
    return s2

def Matern(X):
    s2 = np.zeros((X.shape[0], X.shape[0]))
    l1 = 2
    l2 = 2
    for i in range(X.shape[0]):
        xi = X[i]
        for j in range(X.shape[0]):
            xj = X[j]
            r = np.linalg.norm((xi - xj)/np.array([l1, l2]))
            s2[i, j] = (1 + np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r)   
    return s2
    
def plot_true_obj(n):
    
    fig, axs = plt.subplots(4,2, figsize=(3,2.5), subplot_kw={'projection': '3d'}, dpi=150)
    fig.tight_layout()

    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    x_p, y_p  = np.meshgrid(x,y)

    X = np.transpose(np.vstack([x_p.ravel(), y_p.ravel()]))

    #######################################
    #### SQUARED EXPONENTIAL ##############
    #######################################

    s2 = SE2(X, X)
    mu = np.zeros(n*n)

    sample = multivariate_normal.rvs(mu, s2)   

    axs[0,0].plot_surface(x_p, y_p, sample.reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)

    # Kernel plot

    l1 = 2
    l2 = 2

    r = []
    s2 = []

    for i in range(X.shape[0]):
        xi = X[i]
        s2.append(np.exp(-0.5*(np.linalg.norm((xi - np.array([0, 0]))/(np.array([l1, l2]))))**2))

    axs[0,1].plot_surface(x_p, y_p, np.array(s2).reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

    #######################################
    #### MATERN ###########################
    #######################################

    s2 = Matern(X)
    mu = np.zeros(n*n)

    sample = multivariate_normal.rvs(mu, s2)

    axs[1,0].plot_surface(x_p, y_p, sample.reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

    r = []
    s2 = []

    for i in range(X.shape[0]):
        xi = X[i]
        r = np.linalg.norm(xi - np.array([0,0])/np.array([l1, l2]))
        s2.append((1 + np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r))

    axs[1,1].plot_surface(x_p, y_p, np.array(s2).reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

    #######################################
    #### GAMMA-EXP ########################
    #######################################

    s2 = gamma_exp(X, X)
    mu = np.zeros(n*n)

    sample = multivariate_normal.rvs(mu, s2)

    axs[2,0].plot_surface(x_p, y_p, sample.reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

    r = []
    s2 = []
    gamma = 0.1

    for i in range(X.shape[0]):
        xi = X[i]
        r = np.linalg.norm(xi - np.array([0,0])/np.array([l1, l2]))
        s2.append(np.exp(-r**gamma))

    axs[2,1].plot_surface(x_p, y_p, np.array(s2).reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

    #######################################
    #### RATIONAL-QUADRATIC ###############
    #######################################

    s2 = RQ(X, X)
    mu = np.zeros(n*n)

    sample = multivariate_normal.rvs(mu, s2)

    axs[3,0].plot_surface(x_p, y_p, sample.reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

    r = []
    s2 = []
    alpha = 1

    for i in range(X.shape[0]):
        xi = X[i]
        r = np.linalg.norm(xi - np.array([0,0])/np.array([l1, l2]))
        s2.append((1 + r**2/(2*alpha))**(-alpha))

    axs[3,1].plot_surface(x_p, y_p, np.array(s2).reshape(n,n), edgecolor = 'royalblue', alpha = 0.3, lw = 0.1)

 
    #######################
    ##### PLOTS ###########
    #######################

    axs[0,0].set_title("Prior with SE kernel")
    axs[1,0].set_title("Prior with Matérn kernel ")
    axs[2,0].set_title("Prior with gamma exponential kernel ")
    axs[3,0].set_title("Prior with rational quadratic kernel ")
    axs[0,1].set_title("SE kernel")
    axs[1,1].set_title("Matérn kernel")
    axs[2,1].set_title("Gamma exponential kernel")
    axs[3,1].set_title("Rational quadratic kernel")

    axs[0,0].set_axis_off()
    axs[1,0].set_axis_off()
    axs[2,0].set_axis_off()
    axs[3,0].set_axis_off()
    axs[0,1].set_axis_off()
    axs[1,1].set_axis_off()
    axs[2,1].set_axis_off()
    axs[3,1].set_axis_off()

def mean_prediction(X, X_obs, y_obs):
    sigma_G = SE2(X_obs, X_obs)
    sigma_G_star = SE2(X_obs, X)
    sigma_star = SE2(X, X)
    mu_pred = np.transpose(sigma_G_star) @ np.linalg.inv(sigma_G) @ y_obs 
    # print(sigma_star.shape, sigma_G.shape, sig)
    sigma_pred = sigma_star - np.transpose(sigma_G_star) @ np.linalg.inv(sigma_G) @ sigma_G_star
    sigma_pred = np.sqrt(np.diagonal(sigma_pred))
    return mu_pred, sigma_pred


n = 20
plot_true_obj(n)

x = np.linspace(-1,1,n)
y = np.linspace(-1,1,n)
x_p, y_p  = np.meshgrid(x,y)

X = np.transpose(np.vstack([x_p.ravel(), y_p.ravel()]))

s2 = SE2(X, X)
print(s2)

s2 = gamma_exp(X, X)
print(s2)

s2 = RQ(X, X)
print(s2)

plt.show()




# s2 = np.exp(-0.5*((test)))**2





