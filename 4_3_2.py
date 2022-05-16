import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

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


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

m_1 = np.array([0, 1, 2, 3, -1, -2, -3])
m_4 = np.array([0, -1, -2, 1, 2])
m_2 = np.array([0, 1, -1])
m_3 = np.array([0, -1, 1])
x_1 = np.array([18, 37, 52, 68, 3, -8, -25]) * 0.01
x_2 = np.array([21, 63, -18]) * 0.01
x_3 = np.array([25, -9, 48]) * 0.01
x_4 = np.array([21, 7, -7, 35, 48]) * 0.01

x = np.linspace(-3, 3, 50)

b1, a1, s1, e1 = LSM(m_1, x_1, 7)
b2, a2, s2, e2 = LSM(m_2, x_2, 3)
b3, a3, s3, e3 = LSM(m_3, x_3, 3)
b4, a4, s4, e4 = LSM(m_4, x_4, 5)

y1 = x * b1 + a1
y2 = x * b2 + a2
y3 = x * b3 + a3
y4 = x * b4 + a4

fig, ax = plt.subplots()

y_error = [0.01] * 7

plt.scatter(m_1,x_1, marker = '.', color = 'r', label = '1,22 МГц')
plt.scatter(m_2,x_2, marker = '.', color = 'b', label = '3 МГц')
plt.scatter(m_3,x_3, marker = '.', color = 'g', label = '2 МГц')
plt.scatter(m_4,x_4, marker = '.', color = 'm', label = '1 МГц')

plt.plot(x, y1, color = 'r', linewidth = 0.5)
plt.plot(x, y2, color = 'b', linewidth = 0.5)
plt.plot(x, y3, color = 'g', linewidth = 0.5)
plt.plot(x, y4, color = 'm', linewidth = 0.5)

#plt.errorbar(m_1,x_1,xerr = None,yerr = y_error,fmt='.',color='r',capthick=1,elinewidth=0.5,capsize=1,zorder=10)
#plt.errorbar(m_2,x_2,xerr = None,yerr = y_error[0:3],fmt='.',color='b',capthick=1,elinewidth=0.5,capsize=1,zorder=10)
#plt.errorbar(m_3,x_3,xerr = None,yerr = y_error[0:3],fmt='.',color='g',capthick=1,elinewidth=0.5,capsize=1,zorder=10)
#plt.errorbar(m_4,x_4,xerr = None,yerr = y_error[0:5],fmt='.',color='m',capthick=1,elinewidth=0.5,capsize=1,zorder=10)

plt.grid()
plt.legend()
plt.xlabel('m')
plt.ylabel('xm, мм')

l1 = 30 * 10**(-2) * 6400 * 10**(-10) / b1 * 10**9
l2 = 30 * 10**(-2) * 6400 * 10**(-10) / b2 * 10**9
l3 = 30 * 10**(-2) * 6400 * 10**(-10) / b3 * 10**9
l4 = 30 * 10**(-2) * 6400 * 10**(-10) / b4 * 10**9
ls = np.array([l1, l2, l3, l4])
lm = np.mean(ls)
lms = 1/4*sqrt((lm - l1)**2 + (lm - l2)**2 + (lm - l3)**2 + (lm - l4)**2)

sl1 = l1 * sqrt(e1**2 + (200/6400)**2 + (1/30)**2)
sl2 = l2 * sqrt(e2**2 + (200/6400)**2 + (1/30)**2)
sl3 = l3 * sqrt(e3**2 + (200/6400)**2 + (1/30)**2)
sl4 = l4 * sqrt(e4**2 + (200/6400)**2 + (1/30)**2)

print(b1,s1,l1, sl1)
print(b2,s2,l2, sl2)
print(b3,s3,l3, sl3)
print(b4,s4,l4, sl4)
print()
print(lm,lms)

#plt.show()

