import numpy as np


class NewtonPol:
    def __init__(self, x, y):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.pol_length = len(x)
        self.c = np.copy(y)

        for j in range(1, self.pol_length):
            for i in range(n - 1, j - 1, -1):
                self.c[i] = (self.c[i - 1] - self.c[i]) / (self.x[i - j] - self.x[i])

    def inX(self, x):
        p = 0
        for i in range(self.pol_length):
            m = 1
            for j in range(i):
                m *= x - self.x[j]
            p += self.c[i] * m
        return p


n = 5
l = 10000

x = np.linspace(-1, 1, n)
y = np.cos(np.pi * x)
x_ch = np.cos(np.pi * (2 * np.arange(n) + 1) / 2 / n)
y_ch = np.cos(np.pi * x_ch)

x_test = np.linspace(-1, 1, l)


first = NewtonPol(x, y)
m = abs(first.inX(x_test) - np.cos(np.pi * x_test))
print("Количество узлов: " + str(n) + ";  Интервал: " + str(2 / (n - 1)) + "\n\nРавномерная сетка:")
print(str(np.amax(m)) + "    x = " + str(x_test[np.where(m == np.amax(m))]))


second = NewtonPol(x_ch, y_ch)
m = abs(second.inX(x_test) - np.cos(np.pi * x_test))
print("\nСетка Чебышёва:")
print(str(np.amax(m)) + "    x = " + str(x_test[np.where(m == np.amax(m))]))