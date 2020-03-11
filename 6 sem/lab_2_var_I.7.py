import numpy as np
import matplotlib.pyplot as plt


n = int(input("N = "))
x = np.linspace(0, 1, n)
x_model = np.linspace(0, 1, 11)
h = 1 / n

u_0, u_1 = 0, 1
x_split = 1 / np.sqrt(2)


def k_left(x):
    return np.exp(np.sin(np.ones(x.shape) * x_split))

def q_left(x):
    return 2

def k_right(x):
    return np.ones(x.shape)

def q_right(x):
    return 1

def f(x):
    return np.exp(np.ones(x.shape) * x_split)

def k_left_x(x):
    return np.exp(np.sin(x))

def f_x(x):
    return np.exp(x)

lambda_1 = np.sqrt(q_left(x_split) / k_left(x_split))
lambda_2 = np.sqrt(q_right(x_split) / k_right(x_split))
d_1 = f(x_split) / q_left(x_split)
d_2 = f(x_split) / q_right(x_split)
# print("l1 = {}; l2 = {}; d1 = {}; d2 = {}".format(lambda_1, lambda_2, d_1, d_2))

A = np.zeros((4,4))
A[0,0], A[0,1] = 1, 1
A[1,2], A[1,3] = np.exp(lambda_2), np.exp(-lambda_2)
A[2,0], A[2,1], A[2,2], A[2,3] = np.exp(lambda_1 * x_split), np.exp(-lambda_1 * x_split), \
    -np.exp(lambda_2 * x_split), -np.exp(-lambda_2 * x_split)
A[3,0], A[3,1], A[3,2], A[3,3] = k_left(x_split) * lambda_1 * np.exp(lambda_1 * x_split), \
    -k_left(x_split) * lambda_1 * np.exp(-lambda_1 * x_split), -lambda_2 * np.exp(lambda_2 * x_split), \
        lambda_2 * np.exp(-lambda_2 * x_split)

Ab = np.array([-d_1, 1 - d_2, d_2 - d_1, 0])
C = np.linalg.solve(A, Ab)
# print(A)
# print(C)

u_analit = np.zeros(x.shape)

u_analit[x < x_split] = C[0] * np.exp(lambda_1 * x[x < x_split]) + \
    + C[1] * np.exp(-lambda_1 * x[x < x_split]) + d_1
u_analit[x >= x_split] = C[2] * np.exp(lambda_2 * x[x >= x_split]) + \
    + C[3] * np.exp(-lambda_2 * x[x >= x_split]) + d_2

u_analit[0] = 0 ##)
###########


u_calc = np.zeros(x.shape)
u_calc[0] = u_0
u_calc[-1] = u_1

l_beta = next(i for i, val in enumerate(x) if val > x_split)
l_alpha = l_beta - 1

a_alpha = k_left(x[1:l_alpha] + h / 2)
b_alpha = -(a_alpha + k_left(x[1:l_alpha] - h / 2) + q_left(x[1:l_alpha]) * h**2)
c_alpha = k_left(x[1:l_alpha] - h / 2)
d_alpha = -f(x[1:l_alpha]) * h**2

a_beta = k_right(x[l_beta + 1:n - 1] + h / 2)
b_beta = -(a_beta + k_right(x[l_beta + 1:n - 1] - h / 2) + q_right(x[l_beta + 1:n - 1]) * h**2)
c_beta = k_right(x[l_beta + 1:n - 1] - h / 2)
d_beta = -f(x[l_beta + 1:n - 1]) * h**2

p_alpha = np.zeros(x[1:l_alpha].shape)
q_alpha = np.zeros(p_alpha.shape)

p_alpha[0] = - a_alpha[0] / b_alpha[0]
q_alpha[0] = (d_alpha[0] - c_alpha[0] * u_0) / b_alpha[0]
for i in range(1, l_alpha - 1):
    p_alpha[i] = - a_alpha[i] / (b_alpha[i] + c_alpha[i] * p_alpha[i - 1])
    q_alpha[i] = (d_alpha[i] - c_alpha[i] * q_alpha[i - 1]) / (b_alpha[i] + c_alpha[i] * p_alpha[i - 1])


p_beta = np.zeros(x[l_beta + 1:n - 1].shape)
q_beta = np.zeros(p_beta.shape)

p_beta[-1] = - c_beta[-1] / b_beta[-1]
q_beta[-1] = (d_beta[-1] - a_beta[-1] * u_1) / b_beta[-1]
for i in range(1, len(q_beta)):
    p_beta[-i - 1] = - c_beta[-i - 1] / (b_beta[-i - 1] + a_beta[-i - 1] * p_beta[-i])
    q_beta[-i - 1] = (d_beta[-i - 1] - a_beta[-i - 1] * q_beta[-i]) / (b_beta[-i - 1] + a_beta[-i - 1] * p_beta[-i])

#############

u_calc[l_alpha] = (k_left(x[l_alpha]) * q_alpha[-1] + k_right(x[l_beta]) * q_beta[0]) / \
   (k_left(x[l_alpha]) * (1 - p_alpha[-1]) + k_right(x[l_beta]) * (1 - p_beta[0]))
u_calc[l_beta] = u_calc[l_alpha]

#############


for i in range(1, l_alpha):
    u_calc[l_alpha - i] = p_alpha[-i] * u_calc[l_alpha - i + 1] + q_alpha[-i]

for i in range(len(q_beta)):
    u_calc[l_beta + i + 1] = p_beta[i] * u_calc[l_beta + i] + q_beta[i]


# print(u_calc)


############

idx = np.round(np.linspace(0, len(x) - 1, 11)).astype(int)
x_str = "   X  |" + ("%11.1f " * 11) % tuple(x[idx])
u_analit_str = "[u]l  |" + ("%11.4e " * 11) % tuple(u_analit[idx])
u_calc_str = "  ul  |" + ("%11.4e " * 11) % tuple(u_calc[idx])
delta = u_analit[idx] - u_calc[idx]
delta_str = "  " + chr(916) + "l  |" + ("%11.4e " * 11) % tuple(np.abs(delta))
print(x_str)
print(u_analit_str)
print(u_calc_str)
print(delta_str)



###########

u_calc_x = np.zeros(x.shape)
u_calc_x[0] = u_0
u_calc_x[-1] = u_1

a_alpha = k_left_x(x[1:l_alpha] + h / 2)
b_alpha = -(a_alpha + k_left_x(x[1:l_alpha] - h / 2) + q_left(x[1:l_alpha]) * h**2)
c_alpha = k_left_x(x[1:l_alpha] - h / 2)
d_alpha = -f_x(x[1:l_alpha]) * h**2

a_beta = k_right(x[l_beta + 1:n - 1] + h / 2)
b_beta = -(a_beta + k_right(x[l_beta + 1:n - 1] - h / 2) + q_right(x[l_beta + 1:n - 1]) * h**2)
c_beta = k_right(x[l_beta + 1:n - 1] - h / 2)
d_beta = -f_x(x[l_beta + 1:n - 1]) * h**2

p_alpha = np.zeros(x[1:l_alpha].shape)
q_alpha = np.zeros(p_alpha.shape)

p_alpha[0] = - a_alpha[0] / b_alpha[0]
q_alpha[0] = (d_alpha[0] - c_alpha[0] * u_0) / b_alpha[0]
for i in range(1, l_alpha - 1):
    p_alpha[i] = - a_alpha[i] / (b_alpha[i] + c_alpha[i] * p_alpha[i - 1])
    q_alpha[i] = (d_alpha[i] - c_alpha[i] * q_alpha[i - 1]) / (b_alpha[i] + c_alpha[i] * p_alpha[i - 1])


p_beta = np.zeros(x[l_beta + 1:n - 1].shape)
q_beta = np.zeros(p_beta.shape)

p_beta[-1] = - c_beta[-1] / b_beta[-1]
q_beta[-1] = (d_beta[-1] - a_beta[-1] * u_1) / b_beta[-1]
for i in range(1, len(q_beta)):
    p_beta[-i - 1] = - c_beta[-i - 1] / (b_beta[-i - 1] + a_beta[-i - 1] * p_beta[-i])
    q_beta[-i - 1] = (d_beta[-i - 1] - a_beta[-i - 1] * q_beta[-i]) / (b_beta[-i - 1] + a_beta[-i - 1] * p_beta[-i])

#############

u_calc_x[l_alpha] = (k_left_x(x[l_alpha]) * q_alpha[-1] + k_right(x[l_beta]) * q_beta[0]) / \
   (k_left_x(x[l_alpha]) * (1 - p_alpha[-1]) + k_right(x[l_beta]) * (1 - p_beta[0]))
u_calc_x[l_beta] = u_calc_x[l_alpha]

#############

for i in range(1, l_alpha):
    u_calc_x[l_alpha - i] = p_alpha[-i] * u_calc_x[l_alpha - i + 1] + q_alpha[-i]

for i in range(len(q_beta)):
    u_calc_x[l_beta + i + 1] = p_beta[i] * u_calc_x[l_beta + i] + q_beta[i]

############

u_calc_x_str = "ul(x) |" + ("%11.4e " * 11) % tuple(u_calc_x[idx])
print(u_calc_x_str)

# plt.plot(x, u_analit, x, u_calc, x, u_calc_x, [x_split, x_split], [0, 1])
# plt.show()