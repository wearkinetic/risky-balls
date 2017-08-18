#Monte carlo integration of a spherical harmonic ought to yield 0
from vonMisesFisher import uniform, rejection_sample, get_mle, spherical_harmonic
import numpy as np
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

print("\n ---- Testing with weights")
mu_mle, kappa_mle = get_mle(y, weights=np.ones(y.shape[0]) / y.shape[0] )
print("mu_mle   = ", mu_mle)
print("mu_exact = ", mu)
print("kappa_mle   = ", kappa_mle)
print("kappa_exact = ", kappa)
