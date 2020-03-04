import numpy as np
import matplotlib.pyplot as plt

while(True):

    n = int(input("N = "))       # Количество узлов
    a = int(input("A = "))        # Начальные
    b = int(input("B = "))        # условия

    k = int((n - 1) / 10)

    x = np.linspace(0, 1, n).reshape(n, 1)
    y = np.empty((n, 2))
    y[0] = [a, b]
    h = 1 / (n - 1)     # Шаг

    F1 = np.linalg.inv(np.array([[1, 0], [0, 1]]) - h / 2 * np.array([[99, 40], [250, 99]]))
    F2 = np.linalg.inv(np.array([[1, 0], [0, 1]]) - 5 * h / 12 * np.array([[99, 40], [250, 99]]))
    Fn = np.linalg.inv(np.array([[1, 0], [0, 1]]) - 3 * h / 8 * np.array([[99, 40], [250, 99]]))

    def f(y):
        f = np.empty(2)
        f[0] = 99 * y[0] + 250 * y[1]
        f[1] = 40 * y[0] + 99 * y[1]
        return f

    y[1] = (y[0] + h / 2 * f(y[0])) @ F1
    y[2] = (y[1] + h / 12 * (8 * f(y[1]) - f(y[0]))) @ F2

    for i in range(2, n - 1):
        # k1 = f(y[i]) * h / 6
        # k2 = f(y[i] + h / 2 * k1) * h / 6
        # k3 = f(y[i] + h / 2 * k2) * h / 6
        # k4 = f(y[i] + h * k3) * h / 6

        # y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4)       #   Рунге-Кутты 4

        # y[i + 1] = (y[i] + h / 2 * f(y[i])) @ F1              #   Неявный одношаговый Адамса-Моултона

        # y[i + 1] = (y[i] + h / 12 * (8 * f(y[i]) - f(y[i - 1]))) @ F2

        y[i + 1] = (y[i] + h / 24 * (19 * f(y[i]) - 5 * f(y[i - 1]) + f(y[i - 2]))) @ Fn    # порядок точности 4



    c1 = (2 * a - 5 * b) / 20
    c2 = (2 * a + 5 * b) / 20

    y_a = c1 * np.exp(-x) @ np.array([[5, -2]]) \
        + c2 * np.exp(199 * x) @ np.array([[5, 2]])

    print("\nC1 = {}; C2 = {}\n".format(c1, c2))
    print("{:^6s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format("X", "Y1'", "Y1", "Y2'", "Y2", "d", "e"))

    for i in range(11):
        j = i * k
        d = max([abs(y[j, 0] - y_a[j, 0]), abs(y[j, 1] - y_a[j, 1])])
        print("{:2.1f}{:15E}{:15E}{:15E}{:15E}{:15E}{:15E}"
            .format(i / 10, y[j, 0], y_a[j, 0], y[j, 1], y_a[j, 1], d, d / y[j, 0]))

    # plt.plot(x, y_a[:, 0])
    # plt.plot(x, y[:, 0])
    # plt.show()

    # t = np.linspace(0, 2 * np.pi, 100)
    # e = 1 * np.exp(1j * t)

    # def mu(h):
    #     return 24 * (h**3 - h**2) / (9 * h**3 + 19 * h**2 - 5 * h + 1)

    # plt.fill(mu(e).real, mu(e).imag, (0.4, 0.6, 0.8)) 
    # e = 0.7 * np.exp(1j * t)
    # plt.fill(mu(e).real, mu(e).imag, (0.4, 0.6, 0.8)) 
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()

    print("\n")