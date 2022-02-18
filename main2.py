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
def NV_method(s, X_tjm1, args_schemes):
    kappa, sgm, theta, xi, chi, rho = args_schemes

    d = 2
    s0 = 0.5*s
    s1 = np.sqrt(s)
    s2 = np.sqrt(s)

    eta = np.random.normal(0, 1, size=(d+1))
    eta1 = eta[0]
    eta2 = eta[1]
    eta3 = eta[2]

    if eta3 >= 0:
        X_tj = expV0(d*s0,
                  expV1(s1*eta1,
                        expV2(s2*eta2,
                              expV0(s0,
                                    X_tjm1, kappa, sgm, theta, xi, chi, rho),
                              kappa, sgm, theta, xi, chi, rho),
                        kappa, sgm, theta, xi, chi, rho),
                  kappa, sgm, theta, xi, chi, rho)
    else:
        X_tj = expV0(d*s0,
                  expV2(s2*eta2,
                        expV1(s1*eta1,
                              expV0(s0,
                                    X_tjm1, kappa, sgm, theta, xi, chi, rho),
                              kappa, sgm, theta, xi, chi, rho),
                        kappa, sgm, theta, xi, chi, rho),
                  kappa, sgm, theta, xi, chi, rho)

    return X_tj

@jit(nopython=True)
def V0(I, kappa, sgm, theta, xi, chi, rho):
    w1 = I[0]
    w2 = I[1]
    # w3 = I[2]
    w3 = np.maximum(I[2], 0)
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
    # w3 = I[2]
    w3 = np.maximum(I[2], 0)

    v1 = np.zeros(3)
    v1[0] = sgm*np.sqrt(w3)
    v1[1] = 0
    v1[2] = rho*chi*np.sqrt(w3)

    return v1

@jit(nopython=True)
def V2(I, kappa, sgm, theta, xi, chi, rho):
    w1 = I[0]
    w2 = I[1]
    # w3 = I[2]
    w3 = np.maximum(I[2], 0)

    v2 = np.zeros(3)
    v2[0] = 0
    v2[1] = 0
    v2[2] = np.sqrt(1-rho**2) * chi * np.sqrt(w3)

    return v2

@jit(nopython=True)
def EM_method(s, X_tjm1, args_schemes):
    """
    Inputs:
    Description:
        1. EM-method discretization scheme applied to Ito form SDEs eq (7) on p.1152.
        2. The SDEs have a vector field representation.
        3. Since the SDEs are of Ito form, V_tilde is calculated.
        4. The function implements proposition 1 on p.1149.
        5. For j=1, X_tjm1 = X_t0 = x0
    Return:
        A 3D vector of (x,y,z).
    """
    kappa, sgm, theta, xi, chi, rho = args_schemes

    v0 = V0(X_tjm1, kappa, sgm, theta, xi, chi, rho)
    v1 = V1(X_tjm1, kappa, sgm, theta, xi, chi, rho)
    v2 = V2(X_tjm1, kappa, sgm, theta, xi, chi, rho)
    v0_tilde = v0 + 0.5*(v1*v1 + v2*v2)

    eta = np.random.normal(0, 1, size=(3, 2))
    eta1 = eta[:, 0]
    # print("eta: \n", eta)
    # print("eta1:", eta1)

    # Ito-form
    X_tj = X_tjm1 + (v0_tilde * s) + np.sqrt(s)*v1*eta1 + np.sqrt(s)*v2*eta1

    return X_tj

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
    libor_rate = (1/delta)*(1/P - 1)
    return libor_rate

@jit(nopython=True)
def MC_bond_price(M, discretization_method, discretization_steps, T1, T2, s, X_tjm1, args_schemes, observed_bond_price):
    """
    Description:
        1. Implementing the Monte Carlo method to pricing zero bond price.
        2. The zero bond price is calculated at calendar points T_i i=[1,2,...,K].
        3. Discretisation schemes can be set to EM-scheme or NV-scheme.
    Return:
        Monte Carlo bond price, Monte Carlo standard error
    """
    P0T1, P0T2 = observed_bond_price
    kappa, sgm, theta, xi, chi, rho = args_schemes

    bond_price_arr = np.zeros(M)
    for i in range(M):
        X_arr = np.ones(shape=(3, discretization_steps + 1))
        # print(X_arr)
        for j in range(1, discretization_steps + 1):
            """ Do discretization method """
            X_arr[:, 0] = X_tjm1
            # print("X_tjm1", "j=", j, X_arr[:, j-1])
            x_tj = discretization_method(s, X_arr[:, j - 1], args_schemes)
            X_arr[:, j] = x_tj
            # print("X_tj", "j=", j, X_arr[:, j])
        # print(X_arr)
        """ Do bond price """
        x_Ti = X_arr[0, discretization_steps]
        y_Ti = X_arr[1, discretization_steps]
        # print(x_Ti, "\n", y_Ti)
        # print("Ratio: ", P0T2/P0T1)
        temp_zero_bond_price = zero_bond_price(P0T1, P0T2, x_Ti, y_Ti, T1, T2, chi)
        # print(temp_zero_bond_price)
        # print("Bond price: ", temp_zero_bond_price)
        # print(temp_zero_bond_price)
        bond_price_arr[i] = temp_zero_bond_price
        """ End of MC """
    # print(bond_price_arr.shape)
    mc_bond_mean = bond_price_arr.sum() / M
    mc_bond_stderr = bond_price_arr.std() / np.sqrt(M)
    return mc_bond_mean, mc_bond_stderr, bond_price_arr

@jit(nopython=True)
def snowball_coupon(coupon_im1, L_TiTip1, args_snowball):
    """
    Description:
        This function implements the calculation of coupon following eq(14) on p.1157.
    Inputs:
        1. coupon_im1: coupon from last period
        2. L_TiTip1: Libor rate for period T_i to T_{i+1}
        3. f: floor rate (according to author)
        4. k: margin (according to author)
        5. c: cap rate (according to author)
    Return:
        Coupon at time Ti.
    """
    c, f, k = args_snowball
    temp = np.maximum(f, coupon_im1 + k - L_TiTip1)
    coupon_i = np.minimum(c, temp)
    # coupon_i = np.minimum(c, np.maximum(f, coupon_im1 + k - L_TiTip1))
    return coupon_i

@jit(nopython=True)
def mc_snowball_feat_price(M, K, observed_bond_price_K, I, dist_scheme, dist_steps, args_schemes, args_snowball):
    """
    Description:
        1. Implementing simplified snowball pricing based on eq(14) and eq(15).
    Steps:
        1. Creating a timeline with [0,K] tenors
        2. Starting at T1, calculate coupon_T1, move to T2.
        3. Using MC_bond_price routine to calculate M-dimensional bond price.
        4. At T2, Using coupon_T1 as input for coupon_T2, calculate coupon_T2.
        5. Repeat until T_K to obtain coupon T_K.
        6. Calculate the expected value of coupon payment and standard error.
    Return:
        coupon mean, coupon standard error, (M, K)-dimensional simulated coupons
    """
    coupon_arr = np.zeros(shape=(M, K))

    for Ti in range(1, K):
        s = (Ti - (Ti - 1)) / dist_steps
        observed_bond_price = np.array((observed_bond_price_K[Ti - 1], observed_bond_price_K[Ti]))
        # print("Observed bond price for Ti, T_{i+1}:", observed_bond_price)
        T1 = Ti
        T2 = Ti + 1
        _, _, bond_price_arr = MC_bond_price(M, dist_scheme, dist_steps, T1, T2, s, I, args_schemes, observed_bond_price)
        # print("Bond price: ", bond_price_arr)
        l_rate = Libor(T1, T2, bond_price_arr)
        # print("Libor shape", l_rate.shape)
        # print("Libor rate at Ti=", Ti, ":", l_rate.mean())
        # print(Ti-1, coupon_arr[:, Ti-1])
        coupon_arr[:, Ti] = snowball_coupon(coupon_arr[:, Ti - 1], l_rate, args_snowball)
        # print(coupon_arr.shape)
        # print(Ti, coupon_arr[:, Ti])
        ''' End of calculating coupon_i '''

    ''' Calculating (1) expected total coupon payment & (2) standard error '''
    coupon_mean = coupon_arr.mean()
    coupon_std_err = coupon_arr.std() / np.sqrt(M)

    return coupon_mean, coupon_std_err, coupon_arr

