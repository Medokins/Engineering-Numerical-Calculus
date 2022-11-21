import numpy as np
import matplotlib.pyplot as plt

#NOT WORKING

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
    for x in range(N_X + 1):
        for y in range(N_Y + 1):
            ro[x, y] = get_ro(x * DELTA, y * DELTA)
    return ro

def relaksacja_lokalna(omega_l):
    ro = fill_ro()

    for omega in omega_l:
        v = np.zeros((N_X + 1, N_Y + 1))
        v[:, 0] = V_1
        sums = [1]
    
        while True: 
            v[1:N_X-1, 1:N_Y-1] = (1 - omega)*v[1:N_X-1, 1:N_Y-1] + omega * .25 * (v[2:N_X, 1:N_Y-1] + v[:N_X-2, 1:N_Y-1] + v[1:N_X-1, 2:N_Y] + v[1:N_X-1, :N_Y-2] + np.square(DELTA) * ro[1:N_X-1, 1:N_Y-1] / EPSILON)
            v[0, 1:N_Y-1] = v[1, 1:N_Y-1]
            v[N_X, 1:N_Y-1] = v[N_X-1, 1:N_Y-1]

            sums.append(np.sum(np.square(DELTA) * ((.5 * np.square((v[1:N_X, :N_Y-1] - v[:N_X-1, :N_Y-1]) / DELTA)) + .5 * np.square((v[:N_X-1, 1:N_Y] - v[:N_X-1, :N_Y-1]) / DELTA) - ro[:N_X-1, :N_Y-1] * v[:N_X-1, :N_Y-1])))

            if abs((sums[-1] - sums[-2]) / sums[-2]) < TOL: 
                break

        x, y, z = [], [], [];
        for i in range(1, len(v)):
            for j in range(1, len(v[i])):
                z.append(v[i][j]);
                x.append(i)
                y.append(j)

        plt.title(f'1{omega}');
        plt.tricontourf(x, y, z, cmap='bwr', levels=np.linspace(min(z), max(z), 600))
        plt.colorbar(ticks=np.linspace(min(z), max(z), 10))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f'Local V(x,y), omega={omega}.png')
        plt.clf()

relaksacja_lokalna([1])