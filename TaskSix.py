import matplotlib.pyplot as plt
import numpy as np

#решение y = pi^2

y = np.pi**2 + 1e-15
x = 0
n = 55   # начиная с 55 начинает возростать погрешность
h = 5 / n

x_set = np.ndarray(n)
y_set = np.ndarray(n)

def function(x, y):
    return (y - np.pi**2) / (x - 5)**4

for i in range(n):
    Y = y + h * function(x, y)
    y += h * (function(x, y) + function(x, Y)) / 2
    x += h
    x_set[i] = x
    y_set[i] = y
    print(str(x) + " " + str(y))

#plt.plot(x_set, y_set)
#plt.show()
