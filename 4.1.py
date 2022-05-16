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

y = np.array([1017.5, 1058, 733.7, 809.72, 1054.7, 1228.5, 1155, 1302, 716.22, 607.02, 531.86, 495])
L = np.array([103.5, 103.5, 77.7, 77.7, 110.1, 110.1, 120.5, 120.5, 58.7, 58.7, 46.5, 46.5])
y_err = np.array([20.70697876, 2.767207667, 14.94443546, 2.962092955, 22.26948951, 3.03034951, 24.6154166, 2.892658944, 9.337383603, 3.169759366, 5.969107471, 3.840012316])
L_err = 0.5 * np.ones(6)
y1 = np.array([4151.25, 3088.8, 4570.4, 4916.25, 2661.69, 2068.16])
y1_err = np.array([76.43695319, 57.50352409, 87.13849859, 94.43352115, 34.16418473, 26.16119135])


p1, p2, s, e = chi_sq(L[1:12:2], y[1:12:2], y_err[1:12:2])
x = np.linspace(46, 121, 100)
z = p1 * x + p2

#p1, p2, s, e = chi_sq(L[1:12:2], y1, y1_err)
#x = np.linspace(46, 121, 100)
#z = p1 * x + p2

plt.scatter(L[1:12:2], y[1:12:2], color = 'r', marker = '.')
plt.plot(x, z, color = 'r', lw = 0.5)
plt.errorbar(L[1:12:2], y[1:12:2],xerr = L_err,yerr = y_err[1:12:2],fmt='.',color='r',capthick=1,elinewidth=0.5,capsize=1,zorder=10)

#plt.scatter(L[1:12:2], y1, color = 'g', marker = '.')
#plt.plot(x, z, color = 'g', lw = 0.5)
#plt.errorbar(L[1:12:2], y1,xerr = L_err,yerr = y1_err,fmt='.',color='g',capthick=1,elinewidth=0.5,capsize=1,zorder=10)

plt.ylabel('$sL - s^{2}, см^{2}$')
plt.xlabel('$L, см$')

#plt.ylabel('$L^{2} - l^{2}, см^{2}$')
#plt.xlabel('L, см')


#f = 1/4 * sqrt(4*p2 + p1**2)
#delta = p1/2 - 2*f

print(p1, p2)
print(s, e)
#print(f, delta)
plt.show()