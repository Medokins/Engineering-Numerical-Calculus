import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
from ctypes import *

lib = CDLL("./lib.so")
pmgmres_ilu_cr = lib.pmgmres_ilu_cr
pmgmres_ilu_cr.restype = None
pmgmres_ilu_cr.argtypes = [c_int, c_int,
                           ndpointer(c_int, flags="C_CONTIGUOUS"),
                           ndpointer(c_int, flags="C_CONTIGUOUS"),
                           ndpointer(c_double, flags="C_CONTIGUOUS"),
                           ndpointer(c_double, flags="C_CONTIGUOUS"),
                           ndpointer(c_double, flags="C_CONTIGUOUS"), 
                           c_int, c_int,
                           c_double, c_double]

ITR_MAX = 500
MR = 500
TOL_ABS = TOL_REL = 1e-8
 
def get_epsilon_l(n_x, epsilon_1, epsilon_2, i):
    return epsilon_1 if i <= n_x / 2 else epsilon_2

def get_ro_1(delta, i, j, x_max, y_max, sigma):
    return np.exp(-np.square(((delta * i - .25 * x_max) / sigma)) - np.square(((delta * j - .5 * y_max) / sigma)))

def get_ro_2(delta, i, j, x_max, y_max, sigma):
    return-np.exp(-np.square(((delta * i - .75 * x_max) / sigma)) - np.square(((delta * j - .5 * y_max) / sigma)))

def get_i(l, n_x):
    j = np.floor(l / (n_x + 1))
    return l - j * (n_x + 1)

def main(n_x, n_y, delta, epsilon_1, epsilon_2, v_1, v_2, v_3, v_4, ro_inside):
    N = (n_x + 1) * (n_y + 1)
    x_max = delta * n_x
    y_max = delta * n_y
    sigma = x_max / 10

    a = np.zeros((5 * N), dtype=np.double)
    ia = -1 * np.ones((N + 1), dtype=np.int32)
    ja = np.zeros((5 * N), dtype=np.int32)

    V = np.zeros((N), dtype=np.double)
    b = np.zeros((N), dtype=np.double)

    k = -1

    for l in range(N):
        edge = 0
        vb = 0
        j = np.floor(l / (n_x + 1))
        i = l - j * (n_x + 1)

        if i == 0:
            edge = 1
            vb = v_1

        if j == n_y:
            edge = 1
            vb = v_2

        if i == n_x:
            edge = 1
            vb = v_3

        if j == 0:
            edge = 1
            vb = v_4

        if ro_inside:
            b[l] = -(get_ro_1(delta, i, j, x_max, y_max, sigma) + get_ro_2(delta, i, j, x_max, y_max, sigma))
        elif edge == 1:
            b[l] = vb
        else:
            b[l] = 0

        ia[l] = -1
        if l - n_x - 1 >= 0 and edge == 0:
            k += 1
            if ia[l] < 0:
                ia[l] = k
            a[k] = get_epsilon_l(n_x, epsilon_1, epsilon_2, i) / np.square(delta)
            ja[k] = l - n_x - 1

        if l - 1 >= 0 and edge == 0:
            k += 1
            if ia[l] < 0:
                ia[l] = k
            a[k] = get_epsilon_l(n_x, epsilon_1, epsilon_2, i) / np.square(delta)
            ja[k] = l - 1

        k += 1

        if ia[l] < 0:
            ia[l] = k
        if edge == 0:
            a[k] = -(2 * get_epsilon_l(n_x, epsilon_1, epsilon_2, i) + get_epsilon_l(n_x, epsilon_1, epsilon_2, get_i(l + 1, n_x)) +
                     get_epsilon_l(n_x, epsilon_1, epsilon_2, get_i(l + n_x + 1, n_x))) / np.square(delta)
        else:
            a[k] = 1

        ja[k] = l

        if l < N and edge == 0:
            k += 1
            a[k] = get_epsilon_l(n_x, epsilon_1, epsilon_2, get_i(l + 1, n_x)) / np.square(delta)
            ja[k] = l + 1

        if l < N - n_x - 1 and edge == 0:
            k += 1
            a[k] = get_epsilon_l(n_x, epsilon_1, epsilon_2, get_i(l + n_x + 1, n_x)) / np.square(delta)
            ja[k] = l + n_x + 1

    nz_num = k + 1
    ia[N] = nz_num

    pmgmres_ilu_cr(N, nz_num, ia, ja, a, V, b, ITR_MAX, MR, TOL_ABS, TOL_REL)
    plt.imshow(V.reshape(n_x + 1, n_y + 1), cmap='bwr')
    plt.colorbar()

    if not ro_inside:
        plt.savefig(f"n_x=n_y={n_x}.png")
    else:
        plt.savefig(f"eps_1=1,eps_2={epsilon_2}.png")

    plt.clf()

if __name__ == "__main__":
    DELTA = .1
    EPSILON_1 = EPSILON_2 = 1
    V_1 = V_3 = 10
    V_2 = V_4 = -10

    n_x_n_y_values = [50, 100, 200]

    for value in n_x_n_y_values:
        main(value, value, DELTA, EPSILON_1, EPSILON_2, V_1, V_2, V_3, V_4, False)

    N_X = N_Y = 100
    V_1 = V_2 = V_3 = V_4 = 0
    epsilon_2_values = [1, 2, 10]
    for epsilon in epsilon_2_values:
        main(N_X, N_Y, DELTA, EPSILON_1, epsilon, V_1, V_2, V_3, V_4, True)