import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1
DELTA = .1
N_X = 150 + 1
N_Y = 100 + 1
V_1 = 10
V_2 = 0
X_MAX = DELTA * N_X - N_X
Y_MAX = DELTA * N_Y - N_Y
SIGMA_X = DELTA * X_MAX
SIGMA_Y = DELTA * Y_MAX
TOL = 1e-8

def get_ro(x, y):
    return np.exp(-np.square(((x - 0.35 * X_MAX)/ SIGMA_X)) - np.square(((y - 0.5 * Y_MAX) / SIGMA_Y))) - np.exp(-np.square(((x - 0.65 * X_MAX)/ SIGMA_X)) - np.square(((y - 0.5 * Y_MAX) / SIGMA_Y)))

def fill_ro():
    ro = np.zeros((N_X, N_Y))
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
        sums = [1]

        while True:
            vn[1:N_X-1, 1:N_Y-1] = .25 * (vs[2:N_X, 1:N_Y-1] + vs[:N_X-2, 1:N_Y-1] + vs[1:N_X-1, 2:N_Y] + vs[1:N_X-1, :N_Y-2] + np.square(DELTA) * ro[1:N_X-1, 1:N_Y-1] / EPSILON)
            vn[0, 1:N_Y-1] = vn[1, 1:N_Y-1]
            vn[N_X-1, 1:N_Y-1] = vn[N_X - 2, 1:N_Y - 1]

            vs = np.add(np.multiply((1.0 - omega), vs), np.multiply(omega, vn))
            sums.append(np.sum(np.square(DELTA) * ((.5 * np.square((vs[1:, :N_Y-1] - vs[:N_X-1, :N_Y-1]) / DELTA)) + .5 * np.square((vs[:N_X-1, 1:] - vs[:N_X-1, :N_Y-1]) / DELTA) - ro[:N_X-1, :N_Y-1] * vs[:N_X-1, :N_Y-1])))
            if abs((sums[-1] - sums[-2]) / sums[-2]) < TOL:
                break
    
        error = np.zeros((N_X, N_Y))
        error[1:N_X-1, 1:N_Y-1] = (vn[2:, 1:N_Y-1] - 2 * vn[1:N_X-1, 1:N_Y-1] + vn[:N_X-2, 1:N_Y-1]) / np.square(DELTA) \
                                    + (vn[1:N_X-1, 2:] - 2.0 * vn[1:N_X-1, 1:N_Y-1] + vn[1:N_X-1, :N_Y-2]) / np.square(DELTA) \
                                    + ro[1:N_X-1, 1:N_Y-1] / EPSILON
     
        
        # PLOTS
        x, y, z = [], [], [];
        for i in range(1, len(vn)):
            for j in range(1, len(vn[i])):
                z.append(vn[i][j]);
                x.append(i)
                y.append(j)
        
        # Global S = S(it)
        # plt.title(f"Global S = S(it), omega = {omega}")
        # iter = list(range(len(sums)))
        # plt.plot(it, sums, label=f'S(it), omega = {omega}, {len(iter)}')
        # plt.xlim(left=True)
        # plt.xscale('log')
        # plt.legend()

        # Global V(x, y)
        plt.title(f'1{omega}');
        plt.tricontourf(x, y, z, cmap='bwr', levels=np.linspace(min(z), max(z), 600))
        plt.colorbar(ticks=np.linspace(min(z), max(z), 10))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f'Global V(x,y), omega={omega}.png')
        plt.clf()
        
        # # Global sigma(x,y) error
        plt.title(f"Global Error: omega={omega}")
        plt.imshow(error.T, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"Global error omega={omega}.png", bbox_inches='tight', transparent=False)
        plt.clf()

    #plt.savefig("Global S(it).png")

relaksacja_globalna([.6, 1])