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
args_schemes = np.array((kappa, sgm, theta, xi, chi, rho))

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
# s0 = (T/n) * 0.5
# s1 = np.sqrt(T/n)
# s2 = np.sqrt(T/n)
# s_arr = np.array([s0, s1, s2])

# print(expV0(s0, I, kappa, sgm, theta, xi, chi, rho))
# print(expV1(s1, I, kappa, sgm, theta, xi, chi, rho))
# print(expV2(s2, I, kappa, sgm, theta, xi, chi, rho))

'''
Discretization schemes
    1. EM-method
    2. NV-method
'''
EM_X = EM_method(s, I, args_schemes)
NV_X = NV_method(s, I, args_schemes)
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
observed_bond_price = np.array((P0T1, P0T2))

kappa = 0.1
sgm = 0.02
theta = 1.5
xi = 0.01
chi = 0.01  # Volatility of CIR-process
rho = -0.5
args_schemes = np.array((kappa, sgm, theta, xi, chi, rho))

''' (1) NV-method '''
print("____ [START] NV-method:")

# n_NV = 3  # For testing - 3 steps
# n_NV = 2**3  # 8 steps
n_NV = 2**4  # 16 steps
# n_NV = 2**6  # 64 steps

s_NV = (T2 - T1) / n_NV

M = 2        # Testing only
# M = 2**10    # MC runs = 1,024
# M = 2**16    # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**24    # MC runs = 16,777,216

MC_bond_NV_mean, MC_bond_NV_stderr, _ = MC_bond_price(M, NV_method, n_NV, T1, T2, s_NV, I, args_schemes, observed_bond_price)

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
s_EM = (T2 - T1)/n_EM

M = 2  # Testing only
# M = 2**10  # MC runs = 1,024
# M = 2**16  # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**24  # MC runs = 16,777,216

# n_EM_testing = 1
# testing_MC_bond_EM_mean, testing_MC_bond_EM_stderr = MC_bond_price(M, EM_method, n_EM_testing, T1, T2, s_EM, I, args_schemes, observed_bond_price)
# print("For (n, M):", (n_EM_testing, M), ",")
# print("Testing bond price:", testing_MC_bond_EM_mean)
# print("Standard Error:", testing_MC_bond_EM_stderr)
# print(f"Confidence Interval: [{testing_MC_bond_EM_mean - 2*testing_MC_bond_EM_stderr}, "
#       f"{testing_MC_bond_EM_mean + 2*testing_MC_bond_EM_stderr}]")

MC_bond_EM_mean, MC_bond_EM_stderr, _ = MC_bond_price(M, EM_method, n_EM, T1, T2, s_EM, I, args_schemes, observed_bond_price)
print("For (n, M):", (n_EM, M), ",")
print("Expected bond price:", MC_bond_EM_mean)
print("Standard Error:", MC_bond_EM_stderr)
print(f"Confidence Interval: [{MC_bond_EM_mean - 2*MC_bond_EM_stderr}, {MC_bond_EM_mean + 2*MC_bond_EM_stderr}]")

print("____ [END] EM-method:")
print("\n")

'''
Pricing derivatives 
    1. Snowball's coupon feature (according to current paper)
    2. Asian option (similar to the original paper)
'''
print("____ [START] Snowball coupon feature:")

'''  
Parameters for Snowball (provided by author) 
'''
c = 0.1
# k = 0.6
f = 0.01
# c = 2.0   # Testing
k = 0.05  # Testing
args_snowball = np.array((c, f, k))

''' 
Other necessary parameters 
'''
kappa = 0.1
sgm = 0.02
theta = 1.5
xi = 0.01
chi = 0.01  # Volatility of CIR-process
rho = -0.5
args_schemes = np.array((kappa, sgm, theta, xi, chi, rho))

x0 = 0.0  # According to the paper
y0 = 0.0  # According to the paper
z0 = 0.1  # According to the paper
I = np.array((x0, y0, z0))

''' Assuming flat interest rate '''
K = 10  # 10 annual payments according to author
# K = 3  # For testing
rate = 0.01
observed_bond_price_K = np.array([1/(1+rate)**i for i in range(1, K+1)])
print("Observed coupon prices: \n", observed_bond_price_K)

''' (1) NV-method '''
print("____ [START] Snowball - NV-method :")

n_NV = 2**6     # For testing - 3 steps
# n_NV = 2**7  # 128 steps

M = 2**13    # MC runs = 8,192
# M = 2**16    # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**24    # MC runs = 16,777,216

coupon_mean_nv, coupon_std_err_nv, coupon_arr_nv = mc_snowball_feat_price(M, K, observed_bond_price_K, I,  NV_method, n_NV, args_schemes, args_snowball)
print("For (n, M, K):", (n_NV, M, K), ",")
print("Number of NaN in coupon_array:", np.isnan(coupon_arr_nv).sum())
print("Expected snowball's coupon feature price:", coupon_mean_nv)
print("Standard Error:", coupon_std_err_nv)
print(f"Confidence Interval: [{coupon_mean_nv - 2*coupon_std_err_nv}, {coupon_mean_nv + 2*coupon_std_err_nv}]")

print("____ [END] Snowball - NV-method:")

''' (2) EM-method '''
print("____ [START] Snowball - EM-method:")

n_EM = 2**8

# M = 2**13    # MC runs = 8,192
M = 2**16    # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**24    # MC runs = 16,777,216

coupon_mean_em, coupon_std_err_em, coupon_arr_em = mc_snowball_feat_price(M, K, observed_bond_price_K, I,  EM_method, n_EM, args_schemes, args_snowball)
print("For (n, M, K):", (n_EM, M, K), ",")
print("Expected snowball's coupon feature price:", coupon_mean_em)
print("Standard Error:", coupon_std_err_em)
print(f"Confidence Interval: [{coupon_mean_em - 2*coupon_std_err_em}, {coupon_mean_em + 2*coupon_std_err_em}]")

print("____ [END] Snowball - EM-method:")
print("____ [END] Snowball coupon feature:")
