import numpy as np

class Integral:
    def __init__(self, a, b, fun, error):
        self.a = a
        self.b = b
        self.fun = fun
        self.error = error
    
    def newton_cotes_3by8(self):

        def quadrature(n):
            x = np.linspace(self.a, self.b, n)
            sum = 0
            for i in range(n - 1):
                sum += (fun(x[i]) + 3 * fun((2 * x[i] + x[i + 1]) / 3) + 3 * fun((x[i] + 2 * x[i + 1]) / 3) + fun(x[i + 1])) / 8 * (x[i + 1] - x[i])
            return sum
        
        n = 2 * 1
        while abs(quadrature(2 * n) - quadrature(n)) / 15 > self.error:
            n *= 2
        print("Количество узлов Ньютона-Котеса: " + str(n))
        return quadrature(2 * n)
    
    def gauss_2_knots_sin100x(self, fun):

        def x(t):
            return (self.b + self.a) / 2 + (self.b - self.a) / 2 * t / np.pi

        norm = (self.b - self.a) / 2 / np.pi

        sum = -0.00863503 * fun(0.000594286) + 0.0100118 * fun(1.00518)
        return sum




def fun(x):
    return np.sin(100 * x) / (1 + x)

def g(x):
    return 1 / (1 + x)

error = 1e-6

my_i = Integral(0, 1, fun, error)
print("I = " + str(my_i.newton_cotes_3by8()))
print("I = " + str(my_i.gauss_2_knots_sin100x(g)))