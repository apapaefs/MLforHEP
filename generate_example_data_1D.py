import numpy as np
import random
import math

# first let's define our distribution: 
def Gaussian(x,mu,sigma):
    """Calculate the Gaussan distribution"""
    return 1/sigma/np.sqrt(2 * math.pi) * np.exp( - (x - mu)**2 / (2 * sigma**2))


# Generate N Random Gaussian data with mu, sigma, within xmin, xmax with idx
def GenerateGaussianData(mu,sigma,xmin,xmax,N, idx, weight):
    # perform the von Neumann rejection:
    n = 0 # n will be the counter of generated "events"
    # and save the generated masses in an array:
    x = []
    while n < N:
        xi = random.random()*(xmax-xmin) + xmin  
        r = random.random()
        if r < Gaussian(xi,mu,sigma)/ Gaussian(mu,mu,sigma):
            n = n + 1
            x.append([xi])
        else:
            pass
    identifier = [idx]*N
    weights = [weight]*N
    return x, identifier, weights
