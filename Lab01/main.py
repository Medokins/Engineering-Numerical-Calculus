import numpy as np
import matplotlib.pyplot as plt

### ZAD 1 ###

LAMBDA = -1
T = [0, 5]
DELTA_T = [.01, .1, 1.0]

def f(t, y):
    return LAMBDA * y

def exact_solution(x):
    return np.exp(LAMBDA * x)

def eulera(y, delta_t, n):
    y[n + 1] = y[n] + delta_t * f(delta_t*n, y[n])

def RK2(y, delta_t, n):
    k_1 = f(delta_t*n, y[n])
    k_2 = f(delta_t * n + delta_t, y[n] + delta_t * k_1)

    y[n + 1] = y[n] + delta_t / 2 * (k_1 + k_2)

def RK4(y, delta_t, n):
    k_1 = f(delta_t*n, y[n])
    k_2 = f(delta_t*n + delta_t/2, y[n] + delta_t * k_1 / 2)
    k_3 = f(delta_t*n + delta_t/2, y[n] + delta_t * k_2 / 2)
    k_4 = f(delta_t*n + delta_t, y[n] + delta_t*k_3)

    y[n + 1] = y[n] + delta_t / 6 * (k_1 + 2*k_2 + 2*k_3 + k_4)

def solve(method):
    # Analitycznie
    plt.figure(0)
    # Różnica
    plt.figure(1)

    for delta_t in DELTA_T:
        x = np.zeros(int((T[1] - T[0]) / delta_t) + 1)
        y = np.ones(int((T[1] - T[0]) / delta_t) + 1)
        solution = np.ones(int((T[1] - T[0]) / delta_t) + 1)
        difference = np.zeros(int((T[1] - T[0]) / delta_t) + 1)

        for n in range(int((T[1] - T[0]) / delta_t)):
            x[n + 1] = x[n] + delta_t
            solution[n + 1] = exact_solution(x[n + 1])
            method(y, delta_t, n)
            difference[n + 1]= y[n+1] - solution[n+1]

        plt.figure(0)
        plt.plot(x, y, label = delta_t)
        plt.legend()
        plt.title(f"Analitycznie metodą {method.__name__}")

        plt.figure(1)
        plt.plot(x, difference, label = delta_t)
        plt.legend()
        plt.title(f"Różnica {method.__name__}")

    x = np.zeros(int((T[1] - T[0]) / DELTA_T[0]) + 1)
    y = np.ones(int((T[1] - T[0]) / DELTA_T[0]) + 1)
    
    for n in range(int((T[1] - T[0]) / DELTA_T[0])):
        x[n + 1] = x[n] + DELTA_T[0]
        y[n + 1] = exact_solution(x[n + 1])
    
    plt.show()

solve(eulera)
solve(RK2)
solve(RK4)

#### ZAD 2 ####
plt.close("all")

DELTA_T = 1e-4
R = 100
L = .1
C = .001
OMEGA_0 = 1 / np.sqrt(L * C)
T_0 = 2 * np.pi / OMEGA_0
T = [0, 4*T_0]
OMEGA_V = [.5 * OMEGA_0, .8 * OMEGA_0, 1.0 * OMEGA_0, 1.2 * OMEGA_0]

def V(t, omega_v):
	return 10 * np.sin(omega_v * t)

def f(t, Q, I):
    return I

def g(t, Q, I, omega_v):
    return (V(t, omega_v)/L) - (R/L * I) - (Q / (L*C))

def t_n(n):
    return DELTA_T * n

def RRZ():
	for omega_v in OMEGA_V:
		Q = np.zeros(int((T[1] - T[0]) / DELTA_T) + 1)
		I = np.zeros(int((T[1] - T[0]) / DELTA_T) + 1)
		timestamp = np.zeros(int((T[1] - T[0]) / DELTA_T) + 1)

		for n in range(int((T[1] - T[0]) / DELTA_T)):
			k_Q1 = f(t_n(n), Q[n], I[n])
			k_I1 = g(t_n(n), Q[n], I[n], omega_v)

			k_Q2 = f(t_n(n + 1/2), Q[n] + DELTA_T/2 * k_Q1, I[n] + DELTA_T/2 * k_I1)
			k_I2 = g(t_n(n + 1/2), Q[n] + DELTA_T/2 * k_Q1, I[n] + DELTA_T/2 * k_I1, omega_v)

			k_Q3 = f(t_n(n + 1/2), Q[n] + DELTA_T/2 * k_Q2, I[n] + DELTA_T/2 * k_I2)
			k_I3 = g(t_n(n + 1/2), Q[n] + DELTA_T/2 * k_Q2, I[n] + DELTA_T/2 * k_I2, omega_v)

			k_Q4 = f(t_n(n + 1/2), Q[n] + DELTA_T * k_Q3, I[n] + DELTA_T * k_I3)
			k_I4 = g(t_n(n + 1/2), Q[n] + DELTA_T * k_Q3, I[n] + DELTA_T * k_I3, omega_v)

			Q[n+1] = Q[n] + DELTA_T/6 * (k_Q1 + 2*k_Q2 + 2*k_Q3 + k_Q4)
			timestamp[n+1] = t_n(n)
			I[n+1] = I[n] + DELTA_T/6 * (k_I1 + 2*k_I2 + 2*k_I3 + k_I4)

		plt.plot(timestamp, Q, label=f"omega {omega_v / OMEGA_0}")
		
	plt.title(f"Metoda RK4")
	plt.legend()
	plt.xlabel("t")
	plt.ylabel("Q")
	plt.show()

RRZ()