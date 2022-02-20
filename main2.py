import numpy as np
from numba import jit

@jit(nopython=True)
def calc_G(t, T, chi):
    """
    Input:
        1. t is maturity ant time t.
        2. T is maturity at ti,e T.
        3. 0 < t < T
        3. chi is one-dimensional real number provided by author.
    Usage:
        Helper function to calculate G for bond reconstruction formula.
    Return:
        A one dimension real number.
    """
    return (1-np.exp(chi*(T-t)))/chi

@jit(nopython=True)
def calc_J(xi, chi, theta):
    """
    Inputs:
        1. xi, chi and theta are one dimension  real number.
        2. xi, chi and theta are defined in the paper.
    Usage:
        A helper function to calculate a component of NV-method.
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
        3. s = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. Implements V0 from the equation (9) in p.1153
        2. (w1, w2, w3) are (x,y,z) accordingly.
        3. A helper function to calculate a component for NV-method
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
    B = sgm2*(J/(2*kappa) - (w3+J)/(2*kappa-theta))*(np.exp(-2*kappa*s)-np.exp(-kappa*s))
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
        3. s = tk - tkm1; the time steps for discretized tenor.
    Usage:
        1. Implements V0 from the equation (9) in p.1153
        2. (w1, w2, w3) are (x,y,z) accordingly.
        3. A helper function to calculate a component for NV-method
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
        3. s = tk - tkm1; the time steps for discretised tenor.
    Usage:
        1. Implements V0 from the equation (9) in p.1153
        2. (w1, w2, w3) are (x,y,z) accordingly.
        3. A helper function to calculate a component for NV-method
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
    """
    Inputs:
        1. s = t_{k+1} - t_k
        2. X_tjm1 can be interpreted as initial condition when j=1, or as previous step when j>1.
        3. args_schemes =  (kappa, sgm, theta, xi, chi, rho), are parameters needed.
    Descriptions:
        1. This function implements the NV-method on p.1150, proposition 2.
        2. The NV-method is applied to explicit formula of qGSV-CIR model on p.1153, eqn(9).
    Return:
        A three-dimensional vector containing (x(Ti), y(Ti), z(Ti)).
    """
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
        X_tj = expV0(s0,
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
    """
    Input:
        1. I can be interpreted as initial condition or X_{t_{j-1}}
        2. kappa, sigma, theta, xi, chi and rho are one dimensional real number.
    Description:
        1. This function implements the formula in p.1152, section 3.2.
        2. A helper function to calculate a component for EM-method.
    Return:
        A three-dimensional vector containing V0 (x(T_j),y(T_j),z(T_j))
    """
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
    """
    Input:
        1. I can be interpreted as initial condition or X_{t_{j-1}}
        2. kappa, sigma, theta, xi, chi and rho are one dimensional real number.
    Description:
        1. This function implements the formula in p.1152, section 3.2.
        2. A helper function to calculate a component for EM-method.
    Return:
        A three-dimensional vector containing V1 (x(T_j),y(T_j),z(T_j))
    """
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
    """
    Input:
        1. I can be interpreted as initial condition or X_{t_{j-1}}
        2. kappa, sigma, theta, xi, chi and rho are one dimensional real number.
    Description:
        1. This function implements the formula in p.1152, section 3.2.
        2. A helper function to calculate a component for EM-method.
    Return:
        A three-dimensional vector containing V2 (x(T_j),y(T_j),z(T_j))
    """
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
        1. s = t_j+1 - t_j
        2. X_tjm1 can be interpreted as initial condition when j=1 or X at T_{j-1}
            E.g, for j=1, X_tjm1 = X_t0 = x0
    Description:
        1. EM-method discretization scheme applied to Ito form SDEs eq (7) on p.1152.
        2. The SDEs have a vector field representation.
        3. Since the SDEs are of Ito form, V_tilde is calculated.
        4. The function implements proposition 1 on p.1149.
    Return:
        A 3D vector of (x,y,z).
    """
    kappa, sgm, theta, xi, chi, rho = args_schemes

    v0 = V0(X_tjm1, kappa, sgm, theta, xi, chi, rho)
    v1 = V1(X_tjm1, kappa, sgm, theta, xi, chi, rho)
    v2 = V2(X_tjm1, kappa, sgm, theta, xi, chi, rho)
    correction0 = (sgm**2)/4
    correction1 = 0
    correction2 = (chi**2)/4
    correction = np.array([correction0, correction1, correction2])
    v0_tilde = v0 + correction

    eta = np.random.normal(0, 1, size=(3, 2))
    eta1 = eta[:, 0]
    eta2 = eta[:, 1]

    # Ito form
    X_tj = X_tjm1 + (v0_tilde * s) + np.sqrt(s)*v1*eta1 + np.sqrt(s)*v2*eta2

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
        A M-dimensional vector of zero bond price under chosen discretisation schemes.
    """
    G = calc_G(t, T, chi)
    G2 = G**2
    exponent = -(G*x) - (0.5 * G2 * y)
    P_tT = (P0T/P0t) * np.exp(exponent)

    return P_tT

@jit(nopython=True)
def Libor(T_i, T_ip1, P):
    """
    Inputs:
        1. T_i and T_ip1 are sub-periods
        2. P is the zero bond price calculated using bond reconstruction formula.
        3. P is calculated using zero_bond_price().
    Description:
        1. Calculate the LIBOR rate from t to T using bond price from eq(5).
        2. This implements eq (6) on p.1151.
        3. delta(T_i, T_i+1) is the day count of the period [T_i, T_i+1].
    Return:
        A M-dimensional vector of Libor rate.
    """
    delta = T_ip1 -T_i
    libor_rate = (1/delta)*(1/P - 1)

    return libor_rate

@jit(nopython=True)
def MC_bond_price(M, discretization_method, discretization_steps, T1, T2, s, X_tjm1, args_schemes, observed_bond_price):
    """
    Inputs:
        1. M is the number of Monte Carlo simulation.
        2. discretisation_method is NV-method of EM-method
        3. discretisation_steps is the number of partitions between T1 and T2
        4. T1 and T2 and maturity at T1 and maturity at T2 respectively.
        5. s is the discretised time steps, s = t_k+1 - t_k
        6. X_tjm1 is X at time t_{j-1}. When j =0, X_tjm1 is x0.
        7. args_schemes contains all additional parameters needed by EM-method and NV-method.
        8. Observed_bond_price is the zero bond price at time t_0.
    Description:
        1. Implementing the Monte Carlo method to pricing zero bond price.
        2. The zero bond price is calculated at calendar points T_i, i=[1,2,...,K].
        3. In the Monte Carlo simulation, x(y) and y(t) are simulated using NV- or EM-method according to specification.
        4. The simulated x(t) and y(t) are used in calculation of zero bond price.
        5. At each simulation, one zero bond price will be recorded.
        6. Finally, the bond price is calculated as average of all M bond prices.
    Return:
        Monte Carlo bond price, Monte Carlo standard error, M-dimension vector of zero bond price
    """
    P0T1, P0T2 = observed_bond_price
    kappa, sgm, theta, xi, chi, rho = args_schemes

    bond_price_arr = np.zeros(M)
    for i in range(M):
        X_arr = np.ones(shape=(3, discretization_steps + 1))
        for j in range(1, discretization_steps + 1):
            """ Do discretisation method to generate x(t), y(t) """
            X_arr[:, 0] = X_tjm1
            x_tj = discretization_method(s, X_arr[:, j - 1], args_schemes)
            X_arr[:, j] = x_tj
        """ Do bond price """
        x_Ti = X_arr[0, discretization_steps]
        y_Ti = X_arr[1, discretization_steps]
        temp_zero_bond_price = zero_bond_price(P0T1, P0T2, x_Ti, y_Ti, T1, T2, chi)
        bond_price_arr[i] = temp_zero_bond_price
        """ End of MC """
    mc_bond_mean = bond_price_arr.sum() / M
    mc_bond_stderr = bond_price_arr.std() / np.sqrt(M)
    return mc_bond_mean, mc_bond_stderr, bond_price_arr

@jit(nopython=True)
def snowball_coupon(coupon_im1, L_TiTip1, args_snowball):
    """
    Inputs:
        1. coupon_im1: coupon from last period
        2. L_TiTip1: Libor rate for period T_i to T_{i+1}
        3. args_snowball = (c,f,k)
        3. f: floor rate (according to author)
        4. k: margin (according to author)
        5. c: cap rate (according to author)
    Description:
        This function implements the calculation of coupon following eq(14) on p.1157.
    Return:
        A M-dimension vector of coupon at time Ti.
    """
    c, f, k = args_snowball
    temp = np.maximum(f, coupon_im1 + k - L_TiTip1)
    coupon_i = np.minimum(c, temp)
    return coupon_i

@jit(nopython=True)
def mc_snowball_feat_price(M, K, observed_bond_price_K, I, dist_scheme, dist_steps, args_schemes, args_snowball):
    """
    Inputs:
        1. M is number of Monte Carlo simulation
        2. K is number of coupon term
        3. obvserved_bond_price_K is the bond price observed at time t_0 with maturity {T_i}, i = 1,2,...,K.
        4. I can be interpreted as initial condition or X_{t_j-1}, i.e X of previous step.
        5. dist_scheme is either NV- or EM-method
        6. dist_steps is discretisation steps or number of partitions.
        7. args_schemes contains all parameters for NV- or EM-method.
        8. args_snowball contains all parameters to calculate snowball coupon.
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
        T1 = Ti
        T2 = Ti + 1
        _, _, bond_price_arr = MC_bond_price(M, dist_scheme, dist_steps, T1, T2, s, I, args_schemes, observed_bond_price)
        l_rate = Libor(T1, T2, bond_price_arr)
        coupon_arr[:, Ti] = snowball_coupon(coupon_arr[:, Ti - 1], l_rate, args_snowball)
        ''' End of calculating coupon_i '''

    ''' Calculating (1) expected total coupon payment & (2) standard error '''
    coupon_mean = coupon_arr.mean()
    coupon_std_err = coupon_arr.std() / np.sqrt(M)

    return coupon_mean, coupon_std_err, coupon_arr

