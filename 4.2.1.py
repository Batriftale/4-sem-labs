import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from math import sqrt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def chi_sq(x, y, err):
    function = lambda x, a, b: a * x + b
    popt, pcov = curve_fit(function, xdata=x, ydata=y, sigma=err)

    sigma_a = np.sqrt(pcov[0, 0])
    sigma_b = np.sqrt(pcov[1, 1])

    return popt[0], popt[1], sigma_a, sigma_b


r_l = np.array([0.006764651, 0.021539965, 0.036139911, 0.049774726, 0.064582057, 0.078530254, 0.094445582, 0.106620208])
r_d = np.array([0.003263837, 0.014440829, 0.028039503, 0.043195387, 0.057290816, 0.070729403, 0.088488401, 0.099972954, 0.114146001])
m = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

p11, p12, s1, e1 = chi_sq(m, r_d, None)
p21, p22, s2, e2 = chi_sq(m[0:8], r_l, None)

x = np.linspace(1, 9, 50)
y_d = x * p11 + p12
y_l = x * p21 + p22

plt.scatter(m, r_d, color = 'k', marker ='.')
plt.scatter(m[0:8], r_l, color = 'r', marker = '.')
plt.plot(x, y_d, color = 'k', lw = 0.5, label = 'Темные кольца')
plt.plot(x, y_l, color = 'r', lw = 0.5, label = 'Светлые кольца')
plt.xlabel('m')
plt.ylabel('r$^{2}$, мм$^{2}$')
print(p11, s1)
print(p21, s2)
plt.legend()
plt.show()