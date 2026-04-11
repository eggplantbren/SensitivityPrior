import matplotlib.pyplot as plt
import numpy as np


num_params = 3


xs = np.linspace(-10.0, 10.0, 1001)
p0 = np.exp(-0.5*xs**2)/np.sqrt(2.0*np.pi)
p1 = np.exp(-0.5*(xs - 0.1)**2)/np.sqrt(2.0*np.pi)
p2 = np.exp(-0.5*(xs - 1.0)**2/2.0**2)/np.sqrt(2.0*np.pi*2.0**2)
p_x_given_theta = [p0, p1, p2]
h_x_given_theta = []
for i in range(3):
    ps = p_x_given_theta[i]
    h_x_given_theta.append(-np.trapz(ps*np.log(ps + 1E-300), x=xs))

print(h_x_given_theta)

plt.plot(xs, p0)
plt.plot(xs, p1)
plt.plot(xs, p2)
#plt.show()

def prior_transform(us):
    zs = -100.0 + 200.0*us
    zs[0] = 0.0
    ps = np.exp(zs)
    ps /= np.sum(ps) 
    return ps

def log_likelihood(params):
    px = np.zeros(len(xs)) # Marginal
    h_conditional = 0.0

    for i in range(3):
        # Accumulate marginal
        px += params[i]*p_x_given_theta[i]

        # Entropy of conditional
        h_conditional += params[i]*h_x_given_theta[i]

    # Entropy of marginal
    hx = -np.trapz(px*np.log(px + 1E-300), x=xs)

    # Mutual information
    mi = hx - h_conditional
    return mi

def both(us):
    return log_likelihood(prior_transform(us))

