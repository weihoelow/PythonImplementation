from main import *

'''
Parameters
'''
kappa = 0.1
sgm = 0.02
theta = 1.5
xi = 0.01
chi = 0.01
rho = -0.5
delta = 1  # delta = tk -tkm1

'''
Initial conditions
'''
x0 = 0
y0 =0
z0 = 0.1

print(calc_omega1(x0, y0, z0, kappa, sgm, theta, xi, chi, rho, delta))
print(calc_omega2(x0, y0, z0, kappa, sgm, theta, xi, chi, rho, delta))
print(calc_omega3(x0, y0, z0, kappa, sgm, theta, xi, chi, rho, delta))
