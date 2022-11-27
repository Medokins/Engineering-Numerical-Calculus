import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import pandas as pd

DELTA = .2
N_X = 128
N_Y = 128
X_MAX = DELTA * N_X
Y_MAX = DELTA * N_Y
TOL = 1e-8
K = [16, 8, 4, 2, 1]

def vb_1():
    return np.sin(np.pi * DELTA * np.arange( N_Y + 1) / Y_MAX)

def vb_2():
    return -np.sin(2 * np.pi * DELTA * np.arange(N_X + 1) / X_MAX)

def vb_3():
    return np.sin(np.pi * DELTA * np.arange(N_Y + 1) / Y_MAX)

def vb_4():
    return np.sin(2 * np.pi * DELTA * np.arange(N_X + 1) / X_MAX)

def prepare_v(v):
    v[:, N_Y] = vb_2()
    v[:, 0] = vb_4()
    v[N_X, :] = vb_1()
    v[0, :] = vb_3()
    return v

@nb.jit(nopython=True)
def it_1(k, v) -> np.ndarray:
    for i in range(k, N_X, k):
        for j in range(k, N_Y, k):
            v[i, j] = .25 * (v[i + k, j] + v[i - k, j] + v[i, j + k] + v[i, j - k])
    return v

@nb.jit(nopython=True)
def it_2(k, v) -> np.float32:
    s = 0
    for i in range(k, N_X , k):
        for j in range(k, N_Y, k):
            s += .5 * (k * np.square(DELTA)) * \
                      np.square(((v[i + k, j] - v[i, j] + v[i + k, j + k] - v[i, j + k]) / (2.0 * k * DELTA))) + \
                      np.square((v[i, j + k] - v[i, j] + v[i + k, j + k] - v[i + k, j]) / (2.0 * k * DELTA))
    return s

@nb.jit(nopython=True)
def it_3(k, v) -> np.ndarray:
    k2 = int(k / 2)
    if k != 1:
        for i in range(k, N_X, k):
            for j in range(k, N_Y, k):
                v[i + k2, j + k2] = .25 * (v[i, j] + v[i + k, j] + v[i, j + k] + v[i + k, j + k])
                v[i + k, j + k2] = .5 * (v[i + k, j] + v[i + k, j + k])
                v[i + k2, j + k] = .5 * (v[i, j + k] + v[i + k, j + k])
                v[i + k2, j] = .5 * (v[i, j] + v[i + k, j])
                v[i, j + k2] = .5 * (v[i, j] + v[i, j + k])
    return v

def check_TOL(sums: np.ndarray) -> np.bool8:
    return np.fabs((sums[-1] - sums[-2]) / sums[-2]) < TOL

def PoissonMultiGrid():
    v = np.zeros((N_X + 1, N_Y + 1))
    v = prepare_v(v)
    i = np.zeros(2)

    for k in K:
        sums = [1]
        i[0] = i[1]
        while True:
            i[1] += 1
            v = it_1(k, v)
            sums.append(it_2(k, v))
            if check_TOL(sums):
                break

        data_dict = {'i': np.arange(i[0] + 1, i[1] + 1), 'sums': sums[1:]}
        data = pd.DataFrame(data_dict)
        data.to_csv(f'Sk={k}.csv', index=False)

        plt.title(f'k = {k}')
        plt.imshow(v.T, cmap='viridis')
        plt.imshow(np.flip(v[::k, ::k].T, 1), cmap='hot', interpolation='none')
        plt.colorbar()
        plt.savefig(f'heatmap_k={k}.png')
        plt.clf()

        v = it_3(k, v)
        v = prepare_v(v)

if __name__ == '__main__':
    PoissonMultiGrid()