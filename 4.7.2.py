import matplotlib.pyplot as plt
import numpy as np
#from scipy.optimize import curve_fit
from math import sqrt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


#def chi_sq(x, y, err):
 #   function = lambda x, a, b: a * x + b
  #  popt, pcov = curve_fit(function, xdata=x, ydata=y, sigma=err)

    #sigma_a = np.sqrt(pcov[0, 0])
   # sigma_b = np.sqrt(pcov[1, 1])

   # return popt[0], popt[1], sigma_a, sigma_b


def LSM(x, y, n):
    x_y = np.mean(x * y)
    x_ = np.mean(x)
    y_ = np.mean(y)

    x_2 = np.mean(x ** 2)
    y_2 = np.mean(y ** 2)

    b = (x_y - (x_ * y_))/\
        (x_2 - (x_ ** 2))
    sigma_b = 1/sqrt(n) * sqrt((y_2 - (y_ ** 2))/
                               (x_2 - (x_ ** 2)) - b ** 2)
    epsilon = sigma_b / b

    a = y_ - b * x_

    return b, a, sigma_b, epsilon


r = np.array([2.5, 3.5, 4.5, 5.3, 6])**2
r_err = np.ones(5) * 0.1 * 2

n = np.array([1, 2, 3, 4, 5])

plt.scatter(n, r, marker ='.', color ='r')
b, a, s, e = LSM(n, r, 5)

x = np.linspace(1, 5, 50)
y = b * x + a
plt.plot(x, y, color='r', lw = 0.5)
plt.errorbar(n, r, xerr = None,yerr = r_err,fmt='.',color='r',capthick=1,elinewidth=0.5,capsize=1,zorder=10)
print(b,s)
plt.xlabel('n')
plt.ylabel('r$^{2}$, см$^{2}$')
plt.show()
