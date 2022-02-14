from main2 import *

'''
Parameters - Given in the paper
'''
kappa = 0.1
sgm = 0.02
theta = 1.5
xi = 0.01
chi = 0.01
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
s0 = T/(2*n)
s1 = T/np.sqrt(n)
s2 = T/np.sqrt(n)
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
P0t = 0.8
P0T = 0.7
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
print("\n---------- Start of Testing ----------\n")
''' Common parameters '''

T1 = 1  # T_i
T2 = 2  # T_ip1
P0T1 = 0.8
P0T2 = 0.7

''' (1) NV-method '''
n_NV = 10
# M = 3 # Testing only
M = 2**10  # MC runs = 1,024
# M = 2**16  # MC runs = 65,536
# M = 2**24  # MC runs = 16,777,216

bond_price_arr = np.zeros(M)
for i in range(M):
    X_arr = np.ones(shape=(3, n_NV))
    # print(X_arr)
    s0 = (T2 - T1) / (2 * n_NV)
    s1 = (T2 - T1) / np.sqrt(n_NV)
    s2 = (T2 - T1) / np.sqrt(n_NV)
    s_arr = np.array([s0, s1, s2])
    for j in range(1, n_NV):
        """ Do NV-method """
        X_arr[:, 0] = I
        # print(X_arr)
        # print("X_tjm1", "j=", j, X_arr[:, j-1])
        x_tj = NV_method(s_arr, X_arr[:, j-1], kappa, sgm, theta, xi, chi, rho)
        X_arr[:, j] = x_tj
        # print("X_tj", "j=", j, X_arr[:, j])
    # print(X_arr)
    """ Do bond price from NV-method  """
    x_Ti = X_arr[0, n_NV-1]
    y_Ti = X_arr[1, n_NV-1]
    bond_NV = zero_bond_price(P0T1, P0T2, x_Ti, y_Ti, T1, T2, chi)
    # print("Bond price for NV: ", bond_NV)
    bond_price_arr[i] = bond_NV
    """ End of MC for NV-method """

MC_bond_NV_mean = bond_price_arr.sum()/M
MC_bond_NV_stderr = bond_price_arr.std()/np.sqrt(M)
print("NV-method:")
print("For (n, M):", (n_NV, M), ",")
print("Expected bond price:", MC_bond_NV_mean)
print("Standard Error:", MC_bond_NV_stderr)

''' (2) EM-method '''
n_EM = 80
s = (T2 - T1)/n_EM

# M = 1 # Testing only
M = 2**10  # MC runs = 1,024
# M = 2**16  # MC runs = 65,536
# M = 2**24  # MC runs = 16,777,216

bond_price_arr = np.zeros(M)
for i in range(M):
    X_arr = np.ones(shape=(3, n_EM))
    X_arr[:, 0] = I
    # print(X_arr)
    for j in range(1, n_EM):
        """ Do EM-method """
        x_tj = EM_method(s, X_arr[:, j-1], kappa, sgm, theta, xi, chi, rho)
        X_arr[:, j] = x_tj
        # print(X_arr)
    """ Do bond price from EM-method  """
    x_Ti = X_arr[0, n_EM - 1]
    y_Ti = X_arr[1, n_EM - 1]
    bond_EM = zero_bond_price(P0T1, P0T2, x_Ti, y_Ti, T1, T2, chi)
    # print("Bond price for NV: ", bond_NV)
    bond_price_arr[i] = bond_EM
    """ End of MC for EM-method """

print(bond_price_arr.shape)
MC_bond_EM_mean = bond_price_arr.sum()/M
MC_bond_EM_stderr = bond_price_arr.std()/np.sqrt(M)
print("EM-method:")
print("For (n, M):", (n_EM, M), ",")
print("Expected bond price:", MC_bond_EM_mean)
print("Standard Error:", MC_bond_EM_stderr)
