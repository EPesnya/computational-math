import numpy as np
import matplotlib.pyplot as plt

k = 1500
n = 11 * k
a = 1        # Начальные
b = 1        # условия

x = np.linspace(0, 1, n).reshape(n, 1)
y = np.empty((n, 2))
y[0] = [a, b]
h = 1 / (n - 1)     # Шаг

F1 = np.linalg.inv(np.array([[1, 0], [0, 1]]) - h / 2 * np.array([[99, 40], [250, 99]]))
Fn = np.linalg.inv(np.array([[1, 0], [0, 1]]) - 5 * h / 12 * np.array([[99, 40], [250, 99]]))

def f(y):
    f = np.empty(2)
    f[0] = 99 * y[0] + 250 * y[1]
    f[1] = 40 * y[0] + 99 * y[1]
    return f

y[1] = (y[0] + h / 2 * f(y[0])) @ F1

for i in range(1, n - 1):
    # k1 = f(y[i]) * h / 6
    # k2 = f(y[i] + h / 2 * k1) * h / 6
    # k3 = f(y[i] + h / 2 * k2) * h / 6
    # k4 = f(y[i] + h * k3) * h / 6

    # y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4)       #   Рунге-Кутты 4

    # y[i + 1] = (y[i] + h / 2 * f(y[i])) @ F1              #   Неявный одношаговый Адамса

    y[i + 1] = (y[i] + h / 12 * (8 * f(y[i]) - f(y[i - 1]))) @ Fn


c1 = (2 * a - 5 * b) / 20
c2 = (2 * a + 5 * b) / 20

y_a = c1 * np.exp(-x) @ np.array([[5, -2]]) \
    + c2 * np.exp(199 * x) @ np.array([[5, 2]])

print("A = {}; B = {}\nC1 = {}; C2 = {}\n".format(a, b, c1, c2))
print("{:^6s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format("X", "Y1'", "Y1", "Y2'", "Y2", "d", "e"))

for i in range(n):
    d = max([abs(y[i, 0] - y_a[i, 0]), abs(y[i, 1] - y_a[i, 1])])
    print("{:2.2f}{:15E}{:15E}{:15E}{:15E}{:15E}{:15E}"
    .format(x[i, 0], y[i, 0], y_a[i, 0], y[i, 1], y_a[i, 1], d, d / y[i, 0]))

plt.plot(x, y_a[:, 0])
plt.plot(x, y[:, 0])
plt.show()