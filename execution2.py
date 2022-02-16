from main2 import *

'''
Parameters - Given in the paper
'''
kappa = 0.1
sgm = 0.02
theta = 1.5
xi = 0.01
chi = 0.01  # Volatility of CIR-process
rho = -0.5
n = 20     # discretization steps
M = 2**24  # MC runs

'''
Initial conditions
'''
x0 = 0.0  # According to the paper
y0 = 0.0  # According to the paper
z0 = 0.1  # According to the paper
I = np.array((x0, y0, z0))

'''
Implementing discretization schemes
'''
t = 0
T = 1
s = (T-t)/n
s0 = (T/n) * 0.5
s1 = np.sqrt(T/n)
s2 = np.sqrt(T/n)
s_arr = np.array([s0, s1, s2])

# print(expV0(s0, I, kappa, sgm, theta, xi, chi, rho))
# print(expV1(s1, I, kappa, sgm, theta, xi, chi, rho))
# print(expV2(s2, I, kappa, sgm, theta, xi, chi, rho))

'''
Discretization schemes
    1. EM-method
    2. NV-method
'''
EM_X = EM_method(s, I, kappa, sgm, theta, xi, chi, rho)
NV_X = NV_method(s_arr, I, kappa, sgm, theta, xi, chi, rho)
print(f"EM-method simulation of X_tk = {EM_X}")
print(f"NV-method simulation of X_tk = {NV_X}")

'''
Bond reconstruction formula
'''
P0t = 1/1.05
P0T = 1/1.05**2
print("P0T/P0t =", P0T/P0t)

em_x = EM_X[0]
em_y = EM_X[1]
nv_x = NV_X[0]
nv_y = NV_X[1]
# t, T and chi are defined above

bond_price_EM = zero_bond_price(P0t, P0T, em_x, em_y, t, T, chi)
bond_price_NV = zero_bond_price(P0t, P0T, nv_x, nv_y, t, T, chi)
print(f"EM-method bond price: {bond_price_EM}")
print(f"NV-method bond price: {bond_price_NV}")

'''
LIBOR
'''
LIBOR_EM = Libor(t, T, bond_price_EM)
LIBOR_NV = Libor(t, T, bond_price_NV)
print(f"EM-method LIBOR rate: {LIBOR_EM}")
print(f"NV-method LIBOR rate: {LIBOR_NV}")

'''
Testing: Monte Carlo of bond price
    (1) NV-method
    (2) EM-method
'''
print("\n---------- Start of Testing (bond price) ----------\n")

''' Common parameters '''
T1 = 1  # T_i
T2 = 2  # T_ip1
P0T1 = 1/1.05
P0T2 = 1/(1.05**2)

''' (1) NV-method '''
print("____ [START] NV-method:")

kappa = 0.1
sgm = 0.02
theta = 1.5
xi = 0.01
chi = 0.01  # Volatility of CIR-process
rho = -0.5

# n_NV = 3  # For testing - 3 steps
# n_NV = 2**3  # 8 steps
n_NV = 2**4  # 16 steps
# n_NV = 2**6  # 64 steps

s0 = 0.5 * ((T2 - T1) / n_NV)
s1 = np.sqrt((T2 - T1) / n_NV)
s2 = np.sqrt((T2 - T1) / n_NV)
s_arr = np.array([s0, s1, s2])

# M = 1        # Testing only
# M = 2**10    # MC runs = 1,024
M = 2**16    # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**24    # MC runs = 16,777,216

bond_price_arr = np.zeros(M)
for i in range(M):
    X_arr = np.ones(shape=(3, n_NV+1))
    # print(X_arr)
    for j in range(1, n_NV+1):
        """ Do NV-method """
        X_arr[:, 0] = I
        # print("X_tjm1", "j=", j, X_arr[:, j-1])
        x_tj = NV_method(s_arr, X_arr[:, j-1], kappa, sgm, theta, xi, chi, rho)
        X_arr[:, j] = x_tj
        # print("X_tj", "j=", j, X_arr[:, j])
    # print(X_arr)
    """ Do bond price from NV-method  """
    x_Ti = X_arr[0, n_NV]
    y_Ti = X_arr[1, n_NV]
    # print(x_Ti, "\n", y_Ti)
    bond_NV = zero_bond_price(P0T1, P0T2, x_Ti, y_Ti, T1, T2, chi)
    # print("Bond price for NV: ", bond_NV)
    bond_price_arr[i] = bond_NV
    """ End of MC for NV-method """

MC_bond_NV_mean = bond_price_arr.sum()/M
MC_bond_NV_stderr = bond_price_arr.std()/np.sqrt(M)

print("For (n, M):", (n_NV, M), ",")
print("Expected bond price:", MC_bond_NV_mean)
print("Standard Error:", MC_bond_NV_stderr)
print(f"Confidence Interval: [{MC_bond_NV_mean - 2*MC_bond_NV_stderr}, {MC_bond_NV_mean + 2*MC_bond_NV_stderr}]")
print("____ [END] NV-method:")

''' (2) EM-method '''
print("____ [START] EM-method:")

# n_EM = 3  # For testing only
# n_EM = 80
# n_EM = 2**7  # 128
n_EM = 2**8  # 256
s = (T2 - T1)/n_EM

# M = 1  # Testing only
# M = 2**10  # MC runs = 1,024
M = 2**16  # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**24  # MC runs = 16,777,216

bond_price_arr = np.zeros(M)
testing_bond_price_arr = np.zeros(M)
for i in range(M):
    X_arr = np.ones(shape=(3, n_EM+1))
    X_arr[:, 0] = I
    testing_x = EM_method(1, I, kappa, sgm, theta, xi, chi, rho)
    testing_bond_price_arr[i] = zero_bond_price(P0T1, P0T2, testing_x[0], testing_x[1], T1, T2, chi)
    # print(X_arr)
    for j in range(1, n_EM+1):
        """ Do EM-method """
        x_tj = EM_method(s, X_arr[:, j-1], kappa, sgm, theta, xi, chi, rho)
        X_arr[:, j] = x_tj
    # print(X_arr)
    """ Do bond price from EM-method  """
    x_Ti = X_arr[0, n_EM]
    y_Ti = X_arr[1, n_EM]
    # print(x_Ti, "\n", y_Ti)
    bond_EM = zero_bond_price(P0T1, P0T2, x_Ti, y_Ti, T1, T2, chi)
    bond_price_arr[i] = bond_EM
    """ End of MC for EM-method """

''' One step discretisation '''
testing_MC_bond_EM_mean = testing_bond_price_arr.sum()/M
testing_MC_bond_EM_stderr = testing_bond_price_arr.std()/np.sqrt(M)
''' n_EM steps discretisation '''
MC_bond_EM_mean = bond_price_arr.sum()/M
MC_bond_EM_stderr = bond_price_arr.std()/np.sqrt(M)

print("____ EM-method:")
print("For (n, M):", (1, M), ",")
print("Testing bond price:", testing_MC_bond_EM_mean)
print("Standard Error:", testing_MC_bond_EM_stderr)
print(f"Confidence Interval: [{testing_MC_bond_EM_mean - 2*testing_MC_bond_EM_stderr}, "
      f"{testing_MC_bond_EM_mean + 2*testing_MC_bond_EM_stderr}]")

print("For (n, M):", (n_EM, M), ",")
print("Expected bond price:", MC_bond_EM_mean)
print("Standard Error:", MC_bond_EM_stderr)
print(f"Confidence Interval: [{MC_bond_EM_mean - 2*MC_bond_EM_stderr}, {MC_bond_EM_mean + 2*MC_bond_EM_stderr}]")

print("____ [END] EM-method:")

"""
Pricing derivatives 
    1. Snowball (according to current paper)
    2. Asian option (similar to the original paper)
"""

""" 
Parameters for Snowball (provided by author) 
"""
c = 0.1
f = 0.01
k = 0.6

K = 3  # For testing
# K = 10  # according to author

# for i in range(K):
