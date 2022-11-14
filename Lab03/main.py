import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_0 = .01
V_0 = 0
DELTA_T0 = 1
S = .75
P = 2
T_MAX = 40
ALPHA = 5
SIGMA = 1e-10

def f(t, x, v):
    return v

def g(t, x, v):
    return ALPHA * (1 - np.square(x)) * v - x

def Ex(x2, x1):
    return (x2 - x1) / (np.power(2, P) - 1)

def Ev(v2, v1):
    return (v2 - v1) / (np.power(2, P) - 1)

def F(x_n1, x_n, v_n1, v_n, delta_t):
    return x_n1 - x_n - (delta_t / 2) * (f(np.NaN, np.NaN, v_n) + f(np.NaN, np.NaN, v_n1))

def G(x_n1, x_n, v_n1, v_n, delta_t):
    return v_n1 - v_n - (delta_t / 2) * (g(np.NaN, x_n, v_n) + g(np.NaN, x_n1, v_n1))


def solve(method, tol):
    t = 0
    delta_t = DELTA_T0
    x_n = X_0
    v_n = V_0

    n = 0

    data = {"t": np.empty(10000), "delta_t": np.empty(10000), "x_n": np.empty(10000), "v_n": np.empty(10000)}
    data["t"][n] = t
    data["delta_t"][n] = DELTA_T0
    data["x_n"][n] = x_n
    data["v_n"][n] = v_n

    while True:
        # two steps with delta_t
        x2_n1, v2_n1 = method(x_n, v_n, delta_t, ALPHA)
        x2_n2, v2_n2 = method(x2_n1, v2_n1, delta_t, ALPHA)

        # one step with 2*delta_t
        x1_n2, v1_n2 = method(x_n, v_n, 2 * delta_t, ALPHA)

        E_x = Ex(x2_n2, x1_n2)
        E_v = Ev(v2_n2, v1_n2)

        if max(np.fabs(E_x), np.fabs(E_v)) < tol:
            t += 2 * delta_t
            x_n = x2_n2
            v_n = v2_n2

            n += 1
            data["t"][n] = t
            data["delta_t"][n] = delta_t
            data["x_n"][n] = x_n
            data["v_n"][n] = v_n

        if t > T_MAX:
            break

        delta_t *= np.power(((S * tol) / (max(np.fabs(E_x), np.fabs(E_v)))), (1 / (P + 1)))

    data = {x: data[x][:n+1] for x in data.keys()}
    df = pd.DataFrame(data)
    df.to_csv(f"{method.__name__}_TOL_{tol}.csv", index=False)

################# Ex. 1 #################

def trapezoids_method(x_n, v_n, delta_t, ALPHA):
    a = np.empty((2, 2))
    x_n1 = x_n
    v_n1 = v_n

    while True:
        F_value = F(x_n1, x_n, v_n1, v_n, delta_t)
        G_value = G(x_n1, x_n, v_n1, v_n, delta_t) 

        a[0][0] = 1
        a[0][1] = - delta_t / 2
        a[1][0] = - delta_t / 2 * (-2 * ALPHA * x_n1 * v_n1 - 1)
        a[1][1] = 1 - delta_t / 2 * ALPHA * (1 - np.square(x_n1))

        delta_x = (-F_value * a[1][1] + G_value * a[0][1]) / (a[0][0] * a[1][1] - a[0][1] * a[1][0])
        delta_v = (-G_value * a[0][0] + F_value * a[1][0]) / (a[0][0] * a[1][1] - a[0][1] * a[1][0])

        x_n1 += delta_x
        v_n1 += delta_v

        if (np.fabs(delta_x) < SIGMA) and (np.fabs(delta_v) < SIGMA): 
            break

    return x_n1, v_n1


################# Ex. 2 #################

def RK2_method(x_n, v_n, delta_t, ALPHA):
    k_1x = f(np.NaN, np.NaN, v_n)
    k_1v = g(np.NaN, x_n, v_n)

    k_2x = f(np.NaN, np.NaN, v_n + delta_t * k_1v)
    k_2v = g(np.NaN, x_n + delta_t * k_1x, v_n + delta_t * k_1v)

    x_n1 = x_n + delta_t / 2 * (k_1x + k_2x)
    v_n1 = v_n + delta_t / 2 * (k_1v + k_2v)

    return x_n1, v_n1


################# Solutions #################

solve(trapezoids_method, 1e-2)
solve(trapezoids_method, 1e-5)

solve(RK2_method, 1e-2)
solve(RK2_method, 1e-5)

################# Plots #################

plots = [["t", "v_n"], ["t", "x_n"], ["t", "delta_t"], ["x_n", "v_n"]]
# Trapezoids Method
for plot in plots:
    data_1e2 = np.genfromtxt('trapezoids_method_TOL_0.01.csv', delimiter=',', names=True)
    data_1e5 = np.genfromtxt('trapezoids_method_TOL_1e-05.csv', delimiter=',', names=True)
    plt.figure()
    plt.title(f"trapezoids {plot[1]}({plot[0]})")
    plt.plot(data_1e2[plot[0]], data_1e2[plot[1]], label="1e-2")
    plt.plot(data_1e5[plot[0]], data_1e5[plot[1]], label="1e-5")
    plt.legend()
    plt.show()

# RK2 Method
for plot in plots:
    data_1e2 = np.genfromtxt('RK2_method_TOL_0.01.csv', delimiter=',', names=True)
    data_1e5 = np.genfromtxt('RK2_method_TOL_1e-05.csv', delimiter=',', names=True)
    plt.figure()
    plt.title(f"RK2 {plot[1]}({plot[0]})")
    plt.plot(data_1e2[plot[0]], data_1e2[plot[1]], label="1e-2")
    plt.plot(data_1e5[plot[0]], data_1e5[plot[1]], label="1e-5")
    plt.legend()
    plt.show()