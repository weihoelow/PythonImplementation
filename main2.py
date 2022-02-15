import numpy as np
from numba import jit

@jit(nopython=True)
def calc_h(chi, t):
    """
    Assuming one dimension
    """
    return np.exp(-chi*t)

@jit(nopython=True)
def calc_G(t, T, chi):
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
def expV0(s, I, kappa, sgm, theta, xi, chi, rho):
    """
    Inputs:
        1. I: The 3D vector for the initial conditions for (x,y,z)
        2. kappa, sigma, theta, xi, chi, rho and delta are 1D real number; given in the paper
        3. s : delta = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. (omega1, omega2, omega3) are (x,y,z) accordingly.
    Return:
        The 3D vector of exp(sV0)(X)
    """
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]
    J = calc_J(xi, chi, theta)
    sgm2 = sgm**2

    '''Intermediate steps to calculate elements of v0'''
    A = (sgm2*(w3+J))/((2*kappa-theta)*(kappa-theta)) * (np.exp(-theta*s)-np.exp(-kappa*s))
    B = sgm2*((J/2*kappa) - (w3+J)/(2*kappa-theta))*(np.exp(-2*kappa*s)-np.exp(-kappa*s))
    C = (w2 - (sgm2*J)/(2*kappa) - (rho*xi*sgm)/4) * (1-np.exp(-kappa*s))
    D = (w3+J)/(2*kappa-theta)*(np.exp(-theta*s)-np.exp(-2*kappa*s))
    E = (J/(2*kappa))*(1-np.exp(-2*kappa*s))

    v0 = np.zeros(3)
    v0[0] = w1 + A - B/kappa + C/kappa
    v0[1] = w2 + sgm2*(D - E)
    v0[2] = (w3+J)*np.exp(-theta*s) - J

    return v0

@jit(nopython=True)
def expV1(s, I, kappa, sgm, theta, xi, chi, rho):
    """
    Inputs:
        1. I: The 3D vector for the initial conditions for (x,y,z)
        2. kappa, sigma, theta, xi, chi, rho and delta are 1D real number; given in the paper
        3. s : delta = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. (omega1, omega2, omega3) are (x,y,z) accordingly.
    Return:
        The 3D vector of exp(sV1)(X)
    """
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]

    v1 = np.zeros(3)
    v1[0] = (sgm*s/2 + np.sqrt(w3))**2
    v1[1] = w2
    v1[2] = (rho*xi*s/2 + np.sqrt(w3))**2

    return v1

@jit(nopython=True)
def expV2(s, I, kappa, sgm, theta, xi, chi, rho):
    """
    Inputs:
        1. I: The 3D vector for the initial conditions for (x,y,z)
        2. kappa, sigma, theta, xi, chi, rho and delta are 1D real number; given in the paper
        3. s : delta = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. (omega1, omega2, omega3) are (x,y,z) accordingly.
    Return:
        The 3D vector of exp(sV2)(X)
    """
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]

    v2 = np.zeros(3)
    v2[0] = w1
    v2[1] = w2
    v2[2] = (np.sqrt(1-rho**2)*xi*s/2 + np.sqrt(w3))**2

    return v2

@jit(nopython=True)
def NV_method(s_arr, I, kappa, sgm, theta, xi, chi, rho):
    # np.random.seed(0)

    d = 2
    s0 = s_arr[0]
    s1 = s_arr[1]
    s2 = s_arr[2]

    eta = np.random.normal(0, 1, size=(d+1))
    eta1 = eta[0]
    eta2 = eta[1]
    eta3 = eta[2]

    if eta3 >= 0:
        X = expV0(s0,
                  expV1(s1*eta1,
                        expV2(s2*eta2,
                              expV0(s0, I,
                                    kappa, sgm, theta, xi, chi, rho),
                              kappa, sgm, theta, xi, chi, rho),
                        kappa, sgm, theta, xi, chi, rho),
                  kappa, sgm, theta, xi, chi, rho)
    else:
        X = expV0(d*s0,
                  expV2(s2*eta2,
                        expV1(s1*eta1,
                              expV0(s0, I,
                                    kappa, sgm, theta, xi, chi, rho),
                              kappa, sgm, theta, xi, chi, rho),
                        kappa, sgm, theta, xi, chi, rho),
                  kappa, sgm, theta, xi, chi, rho)

    return X

@jit(nopython=True)
def V0(I, kappa, sgm, theta, xi, chi, rho):
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]
    sgm2 = sgm**2

    v0 = np.zeros(3)
    v0[0] = w2 - kappa*w1 - (rho*chi*sgm)/4
    v0[1] = sgm2*w3 - 2*kappa*w2
    v0[2] = theta*(xi-w3) - (chi**2)/4

    return v0

@jit(nopython=True)
def V1(I, kappa, sgm, theta, xi, chi, rho):
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]

    v1 = np.zeros(3)
    v1[0] = sgm*np.sqrt(w3)
    v1[1] = 0
    v1[2] = rho*chi*np.sqrt(w3)

    return v1

@jit(nopython=True)
def V2(I, kappa, sgm, theta, xi, chi, rho):
    w1 = I[0]
    w2 = I[1]
    w3 = I[2]

    v2 = np.zeros(3)
    v2[0] = 0
    v2[1] = 0
    v2[2] = np.sqrt(1-rho**2) * chi * np.sqrt(w3)

    return v2

@jit(nopython=True)
def EM_method(s, I, kappa, sgm, theta, xi, chi, rho):
    """
    Description:
        EM-method discretization scheme applied to Ito form SDEs eq (7) on p.1152.
        The SDEs have a vector field representation.
        Since the SDEs are of Ito form, V_tilde is calculated.
    Return:
        A 3D vector of (x,y,z).
    """
    # np.random.seed(0)

    # s0 = (T/n)/2
    # s = T/n  # T=1, s=1/n

    v0 = V0(I, kappa, sgm, theta, xi, chi, rho)
    v1 = V1(I, kappa, sgm, theta, xi, chi, rho)
    v2 = V2(I, kappa, sgm, theta, xi, chi, rho)
    v0_tilde = v0 + 0.5*(v1*v1 + v2*v2)

    eta = np.random.normal(0, 1, size=(3, 2))
    eta1 = eta[:, 0]
    eta2 = eta[:, 1]
    # print("eta: \n", eta)
    # print("eta1:", eta1)
    # print("eta2:", eta2)

    # # Stratonovich-form
    # X = I + (v0 * s0) + np.sqrt(s) * v1 * eta1 + np.sqrt(s) * v2 * eta2

    # Ito-form
    X = I + (v0_tilde * s) + np.sqrt(s)*v1*eta1 + np.sqrt(s)*v2*eta2

    return X

@jit(nopython=True)
def zero_bond_price(P0t, P0T, x, y, t, T, chi):
    """
    Description:
        Calculating the bond price which follows qG-model.
        This implements eq (5) on p.1151
    Input:
        1. P0t: Observable price for bond maturing at t.
        2. P0T: Observable price for bond maturing at T.
        3. x: First component of X, i.e X[0]
        4. y: Second component of X, i.e X[1]
        5. t, T are the time to maturity; 0 < t0 < t < T.
    Return:
        Zero bond price according to the discretisation schemes.
    """
    G = calc_G(t, T, chi)
    G2 = G**2
    exponent = -(G*x) - (0.5 * G2 * y)
    P_tT = (P0T/P0t) * np.exp(exponent)

    return P_tT

@jit(nopython=True)
def Libor(T_i, T_ip1, P):
    """
    Description:
        Calculate the LIBOR rate from t to T using bond price from eq(5).
        This implements eq (6) on p.1151.
        delta(T_i, T_i+1) is the day count of the period [T_i, T_i+1].
    Inputs:
        1. T_i and T_ip1 are sub-periods
    Return:
         One-dimensional Libor rate
    """
    delta = T_ip1 -T_i
    Libor_rate = (1/delta)*(1/P - 1)
    return Libor_rate
