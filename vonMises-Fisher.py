#!/usr/local/env python
import numpy as np

#TODO: Perhaps you can make this a distirbution that actually generates samples too
def get_MLE(x):
    """ Computes the MLE for a Fisher-Von Mises distribution

    args:
    x -- array like, shape (N,p), represents samples on the there sphere
        where x[i] \in R^p is of unit magnitude

    output:
    mu -- a unit vector in R^n representing the spherical mean
    kappa -- a positive constant reprsenting the spread
    """
    mu = x.sum(axis=0)
    mu /= np.sqrt(np.dot(mu,mu))
    p = x.shape[1]
    N = x.shape[0]
    R_bar = np.linalg.norm(x.sum(axis=0)) / N
    print("R_bar =",R_bar)
    #A(kappa)=R_bar
    kappa_0 = R_bar*(p-R_bar**2) / (1.0-R_bar**2)
    print("kappa_0 =",kappa_0)
    from scipy.special import iv
    A = lambda kappa: iv(p/2.0, kappa) / iv(p/2.0-1.0, kappa)
    root_func = lambda kappa: A(kappa) - R_bar
    from scipy.optimize import fsolve
    kappa = fsolve(root_func,kappa_0)
    print("A(kappa) =", A(kappa))
    return mu, kappa


if __name__ == "__main__":
    x = np.random.randn(100,3)/3.0
    x[:,0] += 1.0
    x_mag = np.sqrt((x**2).sum(axis=1))
    x /= np.tile(x_mag, (3,1)).transpose()
    mu, kappa = get_MLE(x)
    print("mu = " + str(mu))
    print("kappa = " + str(kappa))
