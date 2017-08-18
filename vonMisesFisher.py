#!/usr/local/env python
""" Routines for sampling on spheres """

import numpy as np

def get_mle(x, weights=0):
    """ Computes the MLE for a Fisher-Von Mises distribution

    args:
    x -- array like, shape (N,p), represents samples on the there sphere
        where x[i] \in R^p is of unit magnitude

    kwargs:
    weights -- array, shape (N,) or None.  default:None

    output:
    mu -- a unit vector in R^n representing the spherical mean
    kappa -- a positive constant reprsenting the spread
    """
    if type(weights)==int:
        weights=np.ones(x.shape[0])
    mu = np.einsum('ij,i->j',x,weights)
    mu /= np.sqrt(np.dot(mu,mu))
    p = x.shape[1]
    N = x.shape[0]
    R_bar = np.linalg.norm( np.einsum('ij,i->j',x,weights) ) / weights.sum()
    #A(kappa)=R_bar
    kappa_0 = R_bar*(p-R_bar**2) / (1.0-R_bar**2)
    from scipy.special import iv
    A = lambda kappa: iv(p/2.0, kappa) / iv(p/2.0-1.0, kappa)
    root_func = lambda kappa: A(kappa) - R_bar
    from scipy.optimize import fsolve
    kappa = fsolve(root_func,kappa_0)
    return mu, kappa

def uniform(N,dim):
    """ Generates uniformaly distributed random samples on a sphere
    args:
        N -- int, number of samples
        dim -- int, dimension of embeddings space, e.g. for a 2-sphere dim=3.

    returns:
        samples: array-like, shape (N,dim)
    """
    x = np.random.randn(N,dim)
    mag = np.sqrt( (x**2).sum(axis=1) )
    return x/np.tile(mag, (3,1)).transpose()

def rejection_sample(N,kappa,mu):
    """ draws N random samples from the Fisher-Von Mises distribution

    args:
        N -- int
        kappa -- positive float
        mu -- unit-vector, shape=(dim,)

    returns:
        samples -- numpy array, shape=(N,dim)
    """
    dim = len(mu)
    if N==0:
        return np.zeros((0,dim))
    x = uniform(N,dim)
    vals = np.exp( np.dot(x,mu) * kappa ) / np.exp(kappa)
    # Now we reject some of these samples
    thresholds = np.random.rand(N)
    x = x[vals > thresholds]
    N_rejected = N - len(x)
    #print(N_rejected)
    y = rejection_sample(N_rejected, kappa, mu)
    return np.concatenate([x,y], axis=0)

def spherical_harmonic(q, m, ell):
    """ Extracts the spherical coordinates

    args:
        q -- numpy array, shape (N,dim) or (dim)
        m -- int (order)
        ell -- int, -ell < m < ell

    returns:
        Y^m_ell(q)
    """
    assert( np.abs(ell) >= m )
    assert( m > 0 )
    x = q[:,0]
    y = q[:,1]
    z = q[:,2]
    from scipy.special import lpmv
    return (x + y*1j)*lpmv(m, ell, z)

if __name__=="__main__":
    #Monte carlo integration of a spherical harmonic ought to yield 0
    q = uniform(1000000,3)
    print("monte_carlo_integral = ", spherical_harmonic(q, 1, 1).mean())
    print("Expected result = 0.0")
    mu = q[0]
    kappa = 2.0
    y = rejection_sample(100000,kappa,mu)
    mu_mle, kappa_mle = get_mle(y)
    print("mu_mle   = ", mu_mle)
    print("mu_exact = ", mu)
    print("kappa_mle   = ", kappa_mle)
    print("kappa_exact = ", kappa)
