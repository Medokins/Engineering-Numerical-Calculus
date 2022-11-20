import numpy as np
import time
import matplotlib.pyplot as plt

EPSILON = 1
DELTA = .1
N_X = 150
N_Y = 100
V_1 = 10
V_2 = 0
X_MAX = DELTA * N_X
Y_MAX = DELTA * N_Y
SIGMA_X = DELTA * X_MAX
SIGMA_Y = DELTA * Y_MAX
TOL = 1e-8

def get_ro(x, y):
    return np.exp(-np.square(((x - 0.35 * X_MAX)/ SIGMA_X)) - np.square(((y - 0.5 * Y_MAX) / SIGMA_Y))) - np.exp(-np.square(((x - 0.65 * X_MAX)/ SIGMA_X)) - np.square(((y - 0.5 * Y_MAX) / SIGMA_Y)))

def fill_ro():
    ro = np.zeros((N_X + 1, N_Y + 1))
    for x in range(N_X):
        for y in range(N_Y):
            ro[x, y] = get_ro(x * DELTA, y * DELTA)
    return ro

def relaksacja_globalna(omega_g):
    ro = fill_ro()

    for omega in omega_g:
        vs = np.zeros((N_X, N_Y))
        vn = np.zeros((N_X, N_Y))
        vs[:, 0] = V_1
        vn[:, 0] = V_1
        s_prev = 1
        s_next = 1

        while True:
            vn[1:N_X-1, 1:N_Y-1] = .25 * (vs[2:N_X, 1:N_Y-1] + vs[:N_X-2, 1:N_Y-1] + vs[1:N_X-1, 2:N_Y] + vs[1:N_X-1, :N_Y-2] + np.square(DELTA) * ro[1:N_X-1, 1:N_Y-1] / EPSILON)
            vn[0, 1:N_Y-1] = vn[1, 1:N_Y-1]
            vn[N_X-1, 1:N_Y-1] = vn[N_X - 2, 1:N_Y - 1]

            vs = np.add(np.multiply((1.0 - omega), vs), np.multiply(omega, vn))
            s_prev = s_next
            s_next = 0
            s_next += np.sum(np.square(DELTA) * ((.5 * np.square((vs[1:, :N_Y-1] - vs[:N_X-1, :N_Y-1]) / DELTA)) + .5 * np.square((vs[:N_X-1, 1:] - vs[:N_X-1, :N_Y-1]) / DELTA) - ro[:N_X-1, :N_Y-1] * vs[:N_X-1, :N_Y-1]))

            if abs((s_next - s_prev) / s_prev) < TOL:
                break
    
        error = np.zeros((N_X, N_Y))
        error[1:N_X-1, 1:N_Y-1] = (vn[2:, 1:N_Y-1] - 2.0 * vn[1:N_X-1, 1:N_Y-1] + vn[:N_X-2, 1:N_Y-1]) / DELTA**2 \
                                    + (vn[1:N_X-1, 2:] - 2.0 * vn[1:N_X-1, 1:N_Y-1] + vn[1:N_X-1, :N_Y-2]) / DELTA ** 2 \
                                    + ro[1:N_X-1, 1:N_Y-1] / EPSILON
     
        plt.title(f"Global Error: omega={omega}")
        plt.imshow(error.T, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"Global error omega={omega}.png", bbox_inches='tight', transparent=False)
        plt.clf()

        plt.title(f"Global: omega={omega}")
        plt.imshow(vn.T, cmap='viridis')
        plt.colorbar()
        plt.savefig(f"Global omega={omega}.png", bbox_inches='tight', transparent=False)
        plt.clf()

def relaksacja_lokalna(omega_l):
    ro = fill_ro()

    for omega in omega_l:
        v = np.zeros((N_X, N_Y))
        v[:, 0] = V_2
        s_prev = np.float64(1)
        s_next = np.float64(1)

        while True:
            v[1:N_X-1, 1:N_Y-1] = (1 - omega)*v[1:N_X-1, 1:N_Y-1] + omega * .25 * (v[2:N_X, 1:N_Y-1] + v[:N_X-2, 1:N_Y-1] + v[1:N_X-1, 2:N_Y] + v[1:N_X-1, :N_Y-2] + np.square(DELTA) * ro[1:N_X-1, 1:N_Y-1] / EPSILON)
            v[0, 1:N_Y-1] = v[1, 1:N_Y-1]
            v[N_X-1, 1:N_Y-1] = v[N_X-2, 1:N_Y-1]

            s_prev = s_next
            s_next = np.float64(0)
            s_next += np.sum(np.square(DELTA) * ((.5 * np.square((v[1:N_X, :N_Y-1] - v[:N_X-1, :N_Y-1]) / DELTA)) + .5 * np.square((v[:N_X-1, 1:N_Y] - v[:N_X-1, :N_Y-1]) / DELTA) - ro[:N_X-1, :N_Y-1] * v[:N_X-1, :N_Y-1]))

            print(abs((s_next - s_prev) / s_prev))
            if abs((s_next - s_prev) / s_prev) < TOL: 
                break


#relaksacja_globalna([.6, 1])
#relaksacja_lokalna([1, 1.4, 1.8, 1.9])