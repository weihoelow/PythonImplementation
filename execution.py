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

EM_X = EM_method(I, kappa, sgm, theta, xi, chi, rho, T, n)
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

bond_price_EM = bond_recon_formula(P0t, P0T, em_x, em_y, t, T, chi)
bond_price_NV = bond_recon_formula(P0t, P0T, nv_x, nv_y, t, T, chi)
print(f"EM-method bond price: {bond_price_EM}")
print(f"NV-method bond price: {bond_price_NV}")





# v0 = V0(I, kappa, sgm, theta, xi, chi, rho)
# v1 = V1(I, kappa, sgm, theta, xi, chi, rho)
# v2 = V2(I, kappa, sgm, theta, xi, chi, rho)
#
# vv1 = v1 * v1
# vv2 = v2 * v2
#
# temp = v0 + 0.5*(vv1 + vv2)
# print(temp)
