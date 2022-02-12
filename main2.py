import numpy as np
from numba import jit

@jit(nopython=True)
def calc_h(chi,t):
    """
    Assuming one dimension
    """
    return np.exp(-chi*t)

@jit(nopython=True)
def calc_G(t,T, chi):
    """
    Assuming one dimension.
    Calculating G to be used as input for bond reconstruction formula
    """
    return (1-np.exp(chi*(T-t)))/chi

@jit(nopython=True)
def calc_J(xi, chi, theta):
    """
    Inputs:
        Inputs are xi, chi and theta which are defined in the paper.
    Usage:
        Calculate a component of NV-method.
    Return:
        A single number.
    """
    return xi - (chi**2)/(4*theta)

@jit(nopython=True)
def expV0(I, kappa, sgm, theta, xi, chi, rho, delta):
    """
    Inputs:
        1. I: The 3D vector for the initial conditions for (x,y,z)
        2. kappa, sigma, theta, xi, chi, rho and delta are 1D real number; given in the paper
        3. s : delta = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. (omega1, omega2, omega3) are (x,y,z) accordingly.
    Return:
    """
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]
    J = calc_J(xi, chi, theta)
    sgm2 = sgm**2
    s = delta
    v0 = np.zeros(3)

    '''Intermediate steps to calculate elements of v0'''
    A = (sgm2*(w3+J))/((s*kappa-theta)*(kappa-theta)) * (np.exp(-theta*s)-np.exp(-kappa*s))
    B = sgm2*(J/2*kappa - (w3+J)/(2*kappa-theta))*(np.exp(-2*kappa*s)-np.exp(-kappa*s))
    C = (w2 - (sgm2*J)/(2*kappa) - (rho*xi*sgm)/4) * (1-np.exp(-kappa*s))
    D = (w3+J)/(2*kappa-theta)*(np.exp(-theta*s)-np.exp(-2*kappa*s))
    E = J*(1-np.exp(-2*kappa*s))/(2*kappa)

    v0[0] = w1 + A - B/kappa + C/kappa
    v0[1] = w2 + sgm2* (D - E)
    v0[2] = (w3+J)*np.exp(-theta*s) - J

    return v0

@jit(nopython=True)
def expV1(I, kappa, sgm, theta, xi, chi, rho, delta):
    """
    Inputs:
        1. I: The 3D vector for the initial conditions for (x,y,z)
        2. kappa, sigma, theta, xi, chi, rho and delta are 1D real number; given in the paper
        3. s : delta = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. (omega1, omega2, omega3) are (x,y,z) accordingly.
    Return:
    """
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]
    s = delta

    v1 = np.zeros(3)
    v1[0] = (sgm*s/2 + np.sqrt(w3))**2
    v1[1] = w2
    v1[2] = (rho*xi*s/2 + np.sqrt(w3))**2

    return v1

@jit(nopython=True)
def expV2(I, kappa, sgm, theta, xi, chi, rho, delta):
    """
    Inputs:
        1. I: The 3D vector for the initial conditions for (x,y,z)
        2. kappa, sigma, theta, xi, chi, rho and delta are 1D real number; given in the paper
        3. s : delta = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. (omega1, omega2, omega3) are (x,y,z) accordingly.
    Return:
    """
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]
    s = delta

    v2 = np.zeros(3)
    v2[0] = w1
    v2[1] = w2
    v2[2] = (np.sqrt(1-rho**2)*xi*s/2 + np.sqrt(w3))**2

    return v2



