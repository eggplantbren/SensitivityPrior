import numpy as np


num_params = 3

def prior_transform(us):
    zs = -100.0 + 200.0*us
    zs[0] = 0.0
    ps = np.exp(zs)
    ps /= np.sum(ps) 
    return ps

def log_likelihood(params):
    h = -np.sum(params*np.log(params))
    return h

def both(us):
    return log_likelihood(prior_transform(us))

