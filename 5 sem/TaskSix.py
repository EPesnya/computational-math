import matplotlib.pyplot as plt
import numpy as np

#решение y = pi^2

n = 2   # начиная с 55 начинает возростать погрешность

# x_set = np.ndarray(n)
# y_set = np.ndarray(n)

def function(x, y):
    return (y - np.pi**2) / (x - 5)**4

while n < 2600:
    y = np.pi**2 + 1e-15
    yk = 0
    x = 0
    h = 5 / n

    for i in range(n):
        x = h * i
        Y = y + h * function(x, y)
        y += h * (function(x, y) + function(x, Y)) / 2
        if i < n - 2:
            yk = y
        # x += h
        # x_set[i] = x
        # y_set[i] = y

    print("L = " + str(n) + "; h = " + str(h) + "; delta = " + str(yk))

    if n > 100:
        n += 100
    else:
        n += 10

# plt.plot(x_set, y_set)
# plt.show()
