import numpy as np
import matplotlib.pyplot as plt

n = 1
a = 5
b = -2

x = np.linspace(0, 1, 11 * n)
y = np.empty((11 * n, 2))
y[0] = [a, b]
h = 1 / (11 * n - 1)

def f(y):
    f = np.empty(2)
    f[0] = 99 * y[0] + 250 * y[1]
    f[1] = 40 * y[0] + 99 * y[1]
    return f

for i in range(11 * n - 1):
    k1 = f(y[i]) * h / 6
    k2 = f(y[i] + h / 2 * k1) * h / 6
    k3 = f(y[i] + h / 2 * k2) * h / 6
    k4 = f(y[i] + h * k3) * h / 6

    y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4)


c1 = (2 * a - 5 * b) / 20
c2 = (2 * a + 5 * b) / 20
print("A = {}; B = {}\nC1 = {}; C2 = {}\n".format(a, b, c1, c2))

y_a = c1 * np.exp(-x).reshape(11 * n, 1) @ np.array([5, -2]).reshape(1, 2) \
    + c2 * np.exp(199 * x).reshape(11 * n, 1) @ np.array([5, 2]).reshape(1, 2)

print("{:^6s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format("X", "Y1'", "Y1", "Y2'", "Y2", "d"))
for i in range(11 * n):
    print("{:2.2f}{:15e}{:15E}{:15E}{:15E}{:15E}"
    .format(x[i], y[i, 0], y_a[i, 0], y[i, 1], y_a[i, 1],\
        max([abs(y[i, 0] - y_a[i, 0]), abs(y[i, 1] - y_a[i, 1])])))

plt.plot(x, y_a[:, 0])
plt.plot(x, y[:, 0])
plt.show()