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
    return (BETA*N - GAMMA)*u - BETA*np.square(u)

def picardMethod():
    u = np.ones(int(TMAX / DELTA_T))
    for n in range(int(TMAX / DELTA_T) - 1):
        u_prev = u[n]
        for _ in range(MIKRO_MAX):
            u_next = u[n] + (DELTA_T / 2) * (f(np.NaN, u[n]) + f(np.NaN, u_prev))
            if abs(u_next - u_prev) < TOL: break
        u[n + 1] = u_next

    z_t = [N - u[t] for t in range(int(TMAX / DELTA_T))]
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
            u_next = u_prev - (u_prev - u[n] - DELTA_T/2 * (f(np.NaN, u[n]) + f(np.NaN, u[n + 1]))) / (1 - DELTA_T/2 * (ALPHA - 2*BETA*u_prev))
            if abs(u_next - u_prev) < TOL: break
        u[n + 1] = u_next

    z_t = [N - u[t] for t in range(int(TMAX / DELTA_T))]
    x = [i * DELTA_T for i in range(int(TMAX / DELTA_T))]

    plt.plot(x, u, label = "u(t)")
    plt.plot(x, z_t, label = "z(t) = N - u(t)")
    plt.title("Newton")
    plt.legend()
    plt.show()

picardMethod()
newtonMethod()

### ZAD2 ###