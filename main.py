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
def calc_omega1(x_tkm1, y_tkm1, z_tkm1, kappa, sgm, theta, xi, chi, rho, delta):
    """
    Description:
        1. omega1 in the closed-form solution for qGSV-CIR model corresponds
            to the x(t) in CIR volatility structure.
        2. T-t will be discretized into n-steps, resulting in t_k, k=0,1,...,n.0
        3. When k = 0, x_t0 = x0 is the initial condition; x0 := 0.
    Input:
        1. kappa, sigma, theta, xi, chi, rho, x( t_{k-1}), y( t_{k-1} ), z( t_{k-1} ).
        2. s = t_k - t_km1 = delta
    Usage:
        0. V_{i}, i=[0,1,2]
        1. Calculate exp{sV(omega1)} by combining omega1 from  V_{0}, V_{1} and V_{2}.
    Return:
        exp{sV0(omega1)}, exp{sV1(omega1)}, exp{sV2(omega1)}
    """
    J = calc_J(xi, chi, theta)
    s = delta

    """ Calculating components of V0 """
    V0_1 = (sgm**2 * (z_tkm1 + J))/((2*kappa-theta)*(kappa-theta)) * (np.exp(-theta*delta)-np.exp(-kappa*delta))
    V0_2 = (sgm**2/kappa)*(J/(2*kappa) - (z_tkm1+J)/(2*kappa-theta))*(np.exp(-2*kappa*s)-np.exp(-kappa*s))
    V0_3 = (y_tkm1 - (sgm**2*J)/(2*kappa)-(rho*chi*sgm)/4)*(1-np.exp(-kappa*s))/kappa

    x_V0 = x_tkm1 + V0_1 - V0_2 + V0_3
    x_V1 = ( (sgm*s)/2 + np.sqrt(z_tkm1))**2
    x_V2 = x_tkm1

    return x_V0, x_V1, x_V2

@jit(nopython=True)
def calc_omega2(x_tkm1, y_tkm1, z_tkm1, kappa, sgm, theta, xi, chi, rho, delta):
    """
    Description:
        1. omega2 in the closed-form solution for qGSV-CIR model corresponds
            to the y(t) in CIR volatility structure.
        2. T-t will be discretized into n-steps, resulting in t_k, k=0,1,...,n.0
        3. When k = 0, y_t0 = y0 is the initial condition; y0 := 0.
    Input:
        1. kappa, sigma, theta, xi, chi, rho, y( t_{k-1} ), z( t_{k-1} ).
        2. s = t_k - t_km1 = delta
    Usage:
        0. V_{i}, i=[0,1,2]
        1. Calculate exp{sV(omega2)} by combining omega2 from  V_{0}, V_{1} and V_{2}.
    Return:
        exp{sV0(omega2)}, exp{sV1(omega2)}, exp{sV2(omega2)}
    """
    J = calc_J(xi, chi, theta)
    s = delta

    V0 = y_tkm1 + sgm**2 * ((z_tkm1+J)/(2*kappa-theta))*\
           (np.exp(-theta*s)-np.exp(-2*kappa*s))\
           *(1-np.exp(-kappa*s))
    V1 = y_tkm1
    V2 = y_tkm1

    y_tk = V0 + V1 + V2

    return V0, V1, V2

@jit(nopython=True)
def calc_omega3(x_tkm1, y_tkm1, z_tkm1, kappa, sgm, theta, xi, chi, rho, delta):
    """
    Description:
        1. omega3 in the closed-form solution for qGSV-CIR model corresponds
            to the y(t) in CIR volatility structure.
        2. T-t will be discretized into n-steps, resulting in t_k, k=0,1,...,n.0
        3. When k = 0, z_t0 = z0 is the initial condition; z0 := 0.01.
    Input:
        1. kappa, sigma, theta, xi, chi, rho, y( t_{k-1} ), z( t_{k-1} ).
        2. s = t_k - t_km1 = delta
    Usage:
        0. V_{i}, i=[0,1,2]
        1. Calculate exp{sV(omega3)} by combining omega3 from  V_{0}, V_{1} and V_{2}.
    Return:
        exp{sV0(omega3)}, exp{sV1(omega3)}, exp{sV2(omega3)}
    """
    J = calc_J(xi, chi, theta)
    s = delta

    V0 = (z_tkm1+J)*np.exp(-theta*s)-J
    V1 = ((rho*chi*s)/2 + np.sqrt(z_tkm1))**2
    V2 = (np.sqrt((1-rho**2)*chi*s)/2 + np.sqrt(z_tkm1))**2

    z_tk = V0 + V1 + V2

    return V0, V1, V2

@jit(nopython=True)
def EM_method(X_tkm1, V0, V1, V2, delta):
    """
    Input:
        1. V0, V1, V2 contains the vector component of x(t), y(t) or z(t) depending on
            which random variable is passed in.
    Usage:
        0. X(t) is the random variable to simulate. X(t) is discretized into n steps s.t {X(t_k)}, k = 1,...,n.
        1. Taking in X(t_km1), apply natural log to X(t_km1).
        2. Simulate eta as standard normal variables ==> N(0,1) for every time step k and dimension i.
            Dimension i follows the index of V.
        3. Simulate the random variable X(t_k) using EM method.
    Return:
        One-dimensional random variable X(t_k)
    """
    """
    Need to adjust for V-tilde
    Need to recode this function 
    """

    d = 2
    s = delta

    print("Input for EM V0:", V0)
    print("Input for EM V1:", V1)
    print("Input for EM V2:", V2)
    # V0 = np.log(V0)/s
    # V1 = np.log(V1)/s
    # V2 = np.log(V2)/s

    eta = np.random.normal(0, 1, size=d)
    eta1 = eta[0]
    eta2 = eta[1]

    X_tk = X_tkm1 + V0*s + V1*eta1*np.sqrt(s) + V2*eta2*np.sqrt(s)
    return X_tk

@jit(nopython=True)
def NV_method(X_tkm1, V0, V1, V2, delta):
    """
    Input:
    Usage:
    Returns:
        One-dimensional random variable X(t_k)
    """
    # np.random.seed(0)
    d = 2
    s = delta

    eta = np.random.normal(0, 1, size=(d + 1))
    eta1 = eta[0]
    eta2 = eta[1]
    eta3 = eta[2]

    print("NV: X_tkm1 = ", X_tkm1)
    print("Eta3 = ", eta3)
    if eta3 >= 0:
        X_tk = np.exp(s*V0/2) * np.exp(np.sqrt(s)*eta1*V1) * np.exp(np.sqrt(s)*eta2*V2) * np.exp(s*V0/2) * X_tkm1
        # X_tk = np.exp(s * V0 / 2) * np.exp(np.sqrt(s) * eta1 * V1) * np.exp(np.sqrt(s) * eta2 * V2) * np.exp(
        #     s * V0 / 2)
    else:
        X_tk = np.exp(d*s*V0/2) * np.exp(np.sqrt(s)*V2*eta2) * np.exp(np.sqrt(s)*eta1*V1) * np.exp(s*V0/2) * X_tkm1
        # X_tk = np.exp(d * s * V0 / 2) * np.exp(np.sqrt(s) * V2 * eta2) * np.exp(np.sqrt(s) * eta1 * V1) * np.exp(
        #     s * V0 / 2)
    print("NV: X_tk = ", X_tk)
    return X_tk





