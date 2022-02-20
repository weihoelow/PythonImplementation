from main2 import *
import matplotlib.pyplot as plt

file1 = open("PythonImplementation-v3.txt", "w")

#---------------------------------------------------------#
#    Section 1: Correctness of NV-method implementation   |
#    Monte Carlo of bond price                            |
#      (1) NV-method                                      |
#      (2) EM-method (benchmark)                          |
#---------------------------------------------------------#

print("\n---------- Start of Testing (bond price) ----------\n", file=file1)
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

x0 = 0.0  # According to the paper
y0 = 0.0  # According to the paper
z0 = 0.1  # According to the paper
I = np.array((x0, y0, z0))

''' (1) NV-method '''
print("____ [START] NV-method:", file=file1)
print("____ [START] NV-method:")

# n_NV = 2**4  # 16 steps
n_NV = 2**6  # 64 steps

s_NV = (T2 - T1) / n_NV

M = 2**10  # MC runs = 1,024
# M = 2**16    # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**24    # MC runs = 16,777,216

MC_bond_NV_mean, MC_bond_NV_stderr, _ = MC_bond_price(M, NV_method, n_NV, T1, T2, s_NV, I, args_schemes, observed_bond_price)

print("For (n, M):", (n_NV, M), ",", file=file1)
print("Expected bond price: ", MC_bond_NV_mean, file=file1)
print("Standard Error:      ", MC_bond_NV_stderr, file=file1)
print(f"Confidence Interval:  [{MC_bond_NV_mean - 2*MC_bond_NV_stderr}, {MC_bond_NV_mean + 2*MC_bond_NV_stderr}]", file=file1)
print("____ [END] NV-method ____", file=file1)
print("____ [END] NV-method ____")

''' (2) EM-method '''
print("____ [START] EM-method:", file=file1)
print("____ [START] EM-method:")

# n_EM = 3  # For testing only
n_EM = 2**8  # 256
s_EM = (T2 - T1)/n_EM

# M = 2  # Testing only
M = 2**10  # MC runs = 1,024
# M = 2**16  # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576

# n_EM_testing = 1
# testing_MC_bond_EM_mean, testing_MC_bond_EM_stderr = MC_bond_price(M, EM_method, n_EM_testing, T1, T2, s_EM, I, args_schemes, observed_bond_price)
# print("For (n, M):", (n_EM_testing, M), ",")
# print("Testing bond price:", testing_MC_bond_EM_mean)
# print("Standard Error:", testing_MC_bond_EM_stderr)
# print(f"Confidence Interval: [{testing_MC_bond_EM_mean - 2*testing_MC_bond_EM_stderr}, "
#       f"{testing_MC_bond_EM_mean + 2*testing_MC_bond_EM_stderr}]")

MC_bond_EM_mean, MC_bond_EM_stderr, _ = MC_bond_price(M, EM_method, n_EM, T1, T2, s_EM, I, args_schemes, observed_bond_price)
print("For (n, M):", (n_EM, M), ",", file=file1)
print("Expected bond price: ", MC_bond_EM_mean, file=file1)
print("Standard Error:      ", MC_bond_EM_stderr, file=file1)
print(f"Confidence Interval: [{MC_bond_EM_mean - 2*MC_bond_EM_stderr}, {MC_bond_EM_mean + 2*MC_bond_EM_stderr}]", file=file1)

print("____ [END] EM-method ____", file=file1)
print("____ [END] EM-method ____")
print("\n---------- End of Testing (bond price) ----------\n", file=file1)
print("\n---------- End of Testing (bond price) ----------\n")

print("\n", file=file1)

#-------------------------------------------------#
#       Section 2: Pricing snowball's coupon      |
#        1. NV-method                             |
#        2. EM-method (benchmark)                 |
#-------------------------------------------------#

print("____ [START] Snowball coupon feature:", file=file1)
print("____ [START] Snowball coupon feature:")

'''  
Parameters for Snowball (provided by author) 
'''
c = 0.1    # according to paper
# k = 0.6    # according to paper
f = 0.01   # according to paper
# c = 2.0    # own parameter
k = 0.05   # own parameter
args_snowball = np.array((c, f, k))
print(f"(c, f, k): {args_snowball}", file=file1)

''' 
Parameters for NV- and EM-method
'''
kappa = 0.1
sgm = 0.02
theta = 1.5
xi = 0.01
chi = 0.01  # Volatility of CIR-process
rho = -0.5
args_schemes = np.array((kappa, sgm, theta, xi, chi, rho))
print(f"(kappa, sgm, theta, xi, chi, rho): {args_schemes}", file=file1)

x0 = 0.0  # According to the paper
y0 = 0.0  # According to the paper
z0 = 0.1  # According to the paper
I = np.array((x0, y0, z0))
print(f"(x0, y0, z0): {I}", file=file1)

''' Assuming flat interest rate '''
K = 10  # 10 annual payments according to author
# K = 3  # For testing
rate = 0.01
observed_bond_price_K = np.array([1/(1+rate)**i for i in range(1, K+1)])
print("Observed coupon prices: \n", observed_bond_price_K, file=file1)

''' (1) NV-method '''
print("____ [START] Snowball - NV-method :", file=file1)
print("____ [START] Snowball - NV-method :")

n_NV = 2**6     # For testing - 64 steps

# M = 2**10    # MC runs = 1,024
M = 2**13    # MC runs = 8,192
# M = 2**16    # MC runs = 65,536
# M = 2**23  # MC runs = 8,388,608

coupon_mean_nv, coupon_std_err_nv, coupon_arr_nv = mc_snowball_feat_price(M, K, observed_bond_price_K, I,  NV_method, n_NV, args_schemes, args_snowball)
print("For (n, M, K):", (n_NV, M, K), ",", file=file1)
print("Number of NaN in coupon_array: ", np.isnan(coupon_arr_nv).sum(), file=file1)
print("Expected snowball's coupon:    ", coupon_mean_nv, file=file1)
print("Standard Error:                ", coupon_std_err_nv, file=file1)
print(f"Confidence Interval:            [{coupon_mean_nv - 2*coupon_std_err_nv}, {coupon_mean_nv + 2*coupon_std_err_nv}]", file=file1)

print("____ [END] Snowball - NV-method:", file=file1)
print("____ [END] Snowball - NV-method:")

''' (2) EM-method '''
print("____ [START] Snowball - EM-method:", file=file1)
print("____ [START] Snowball - EM-method:")

n_EM = 2**8

M = 2**4    # MC runs = 8,192
# M = 2**13    # MC runs = 8,192
# M = 2**16    # MC runs = 65,536
# M = 2**20  # MC runs = 1,048,576
# M = 2**23  # MC runs = 8,388,608
# M = 2**24    # MC runs = 16,777,216

coupon_mean_em, coupon_std_err_em, coupon_arr_em = mc_snowball_feat_price(M, K, observed_bond_price_K, I,  EM_method, n_EM, args_schemes, args_snowball)
print("For (n, M, K):", (n_EM, M, K), ",", file=file1)
print("Expected snowball's coupon:  ", coupon_mean_em, file=file1)
print("Standard Error:              ", coupon_std_err_em, file=file1)
print(f"Confidence Interval:          [{coupon_mean_em - 2*coupon_std_err_em}, {coupon_mean_em + 2*coupon_std_err_em}]", file=file1)
coupon_benchmark = coupon_mean_em

print("____ [END] Snowball - EM-method", file=file1)
print("____ [END] Snowball coupon feature ____", file=file1)
print("____ [END] Snowball - EM-method")
print("____ [END] Snowball coupon feature ____")
print("\n", file=file1)


''' Plotting the error '''

print("____ [START] Discretisation Error", file=file1)

n_steps = np.array([8, 16, 32, 64, 128])

M = 2**4  # MC runs = 1,048,576
# M = 2**20  # MC runs = 1,048,576
dist_err = np.zeros(n_steps.shape[0])

for i in range(n_steps.shape[0]):
    coupon_mean_nv, coupon_std_err_nv, coupon_arr_nv = mc_snowball_feat_price(M, K, observed_bond_price_K, I, NV_method,
                                                                              n_steps[i], args_schemes, args_snowball)
    dist_err[i] = coupon_mean_nv - coupon_benchmark
    print("steps:                                   ", n_steps[i], file=file1)
    print("snowball price (NV-method):              ", coupon_mean_nv, file=file1)
    print("Discretisation Error (C_method - C_true):", dist_err[i], file=file1)

fig = plt.subplot()
fig.plot(n_steps, dist_err, label="NV-method")
plt.title(f"Std error of methods, M={M}")
plt.xlabel("Discretisation steps")
plt.ylabel("Discretisation Error")
plt.legend()
plt.show()
