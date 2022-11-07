import numpy as np
import matplotlib.pyplot as plt

BETA = .001
N = 500
GAMMA = .1
ALPHA = BETA * N - GAMMA
TMAX = 100
DELTA_T = .1
U_0 = 1
TOL = 1e-6
MIKRO_MAX = 20

### ZAD1 ###
def f(t,u):
    return (BETA*N - GAMMA) * u - BETA * np.square(u)

def picardMethod():
    u = np.ones(int(TMAX / DELTA_T))
    for n in range(int(TMAX / DELTA_T) - 1):
        u_prev = u[n]
        for _ in range(MIKRO_MAX):
            u_next = u[n] + (DELTA_T / 2) * (f(np.NaN, u[n]) + f(np.NaN, u_prev))
            if np.abs(u_next - u_prev) < TOL:
                break
            u_prev = u_next
        u[n + 1] = u_next

    z_t = [N - u[i] for i in range(int(TMAX / DELTA_T))]
    x = [i * DELTA_T for i in range(int(TMAX / DELTA_T))]

    plt.plot(x, u, label = "u(t)")
    plt.plot(x, z_t, label = "z(t) = N - u(t)")
    plt.title("Picard")
    plt.legend()
    plt.show()

def newtonMethod():
    u = np.ones(int(TMAX / DELTA_T))
    for n in range(int(TMAX / DELTA_T) - 1):
        u_prev = u[n]
        for _ in range(MIKRO_MAX):
            u_next = u_prev - (u_prev - u[n] - DELTA_T/2 * (f(np.NaN, u[n]) + f(np.NaN, u_prev))) / (1 - DELTA_T/2 * (ALPHA - 2*BETA*u_prev))
            if np.abs(u_next - u_prev) < TOL: 
                break
            u_prev = u_next
        u[n + 1] = u_next

    z_t = [N - u[i] for i in range(int(TMAX / DELTA_T))]
    x = [i * DELTA_T for i in range(int(TMAX / DELTA_T))]

    plt.plot(x, u, label = "u(t)")
    plt.plot(x, z_t, label = "z(t) = N - u(t)")
    plt.title("Newton")
    plt.legend()
    plt.show()

### ZAD2 ###

A11 = .25
A12 = .25 - np.sqrt(3)/6
A21 = .25 + np.sqrt(3)/6
A22 = .25

B1 = .5
B2 = .5

def F1(U1, U2, un):
    return U1 - un - DELTA_T * (A11 * f(np.NaN, U1) + A12 * f(np.NaN, U2))

def F2(U1, U2, un):
    return U2 - un - DELTA_T * (A21 * f(np.NaN, U1) + A22 * f(np.NaN, U2))

def RK2Method():
    u = np.ones(int(TMAX / DELTA_T))
    for n in range(int(TMAX / DELTA_T) - 1):
        u1 = u2 = u[n]
        for _ in range(MIKRO_MAX):
            m11 = 1 - DELTA_T * A11 * (ALPHA - 2 * BETA * u1)
            m12 = - DELTA_T * A12 * (ALPHA - 2 * BETA * u2)
            m21 = - DELTA_T * A21 * (ALPHA - 2 * BETA * u1)
            m22 = 1 - DELTA_T * A22 * (ALPHA - 2 * BETA * u2)

            deltaU1 = (F2(u1, u2, u[n]) * m12 - F1(u1, u2, u[n]) * m22) / (m11 * m22 - m12 * m21)
            deltaU2 = (F1(u1, u2, u[n]) * m21 - F2(u1, u2, u[n]) * m11) / (m11 * m22 - m12 * m21)

            u1 += deltaU1
            u2 += deltaU2

            if (np.abs(deltaU1) and np.abs(deltaU2)) < TOL:
                break

        u[n + 1] = u[n] + DELTA_T * (B1 * f(np.NaN, u1) + B2 * f(np.NaN, u2))

    z_t = [N - u[i] for i in range(int(TMAX / DELTA_T))]
    x = [i * DELTA_T for i in range(int(TMAX / DELTA_T))]

    plt.plot(x, u, label = "u(t)")
    plt.plot(x, z_t, label = "z(t) = N - u(t)")
    plt.title("RK2")
    plt.legend()
    plt.show()

picardMethod()
newtonMethod()
RK2Method()