from main import *
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
delta = 1  # s:= delta = tk -tkm1
n = 20  # discretization steps
M = 2**24  # MC runs

'''
Initial conditions
'''
x0 = 0.0  # According to the paper
y0 = 0.0  # According to the paper
z0 = 0.1  # According to the paper
I = np.array((x0, y0, z0))
'''
Execution of discretization scheme
'''
#
# V0_x, V1_x, V2_x = calc_omega1(x0, y0, z0, kappa, sgm, theta, xi, chi, rho, delta)
# print(V0_x, V1_x, V2_x)
#
# # print(calc_omega2(x0, y0, z0, kappa, sgm, theta, xi, chi, rho, delta))
# # print(calc_omega3(x0, y0, z0, kappa, sgm, theta, xi, chi, rho, delta))
#
# print(f"EM method simulation of X_tk = {EM_method(x0, V0_x, V1_x, V2_x, delta)}")
# print(f"NV method simulation of X_tk = {NV_method(x0, V0_x, V1_x, V2_x, delta)}")

print(expV0(I, kappa, sgm, theta, xi, chi, rho, delta))
print(expV1(I, kappa, sgm, theta, xi, chi, rho, delta))
print(expV2(I, kappa, sgm, theta, xi, chi, rho, delta))

