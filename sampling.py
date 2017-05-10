#!/usr/bin/env python
""" Routines for sampling on spheres """

import numpy as np

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

if __name__=="__main__":
    x = uniform(10,3)
    print(x)
    print( (x**2).sum(axis=1))
