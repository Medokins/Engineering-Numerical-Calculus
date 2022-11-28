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
    return np.sin(np.pi * DELTA * np.arange(N_Y + 1) / Y_MAX)

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
def integral(k, v) -> np.float32:
    s = 0
    for i in range(0, N_X , k):
        for j in range(0, N_Y, k):
            s += .5 * (np.square(DELTA * k)) * \
                      (np.square(((v[i + k, j] - v[i, j] + v[i + k, j + k] - v[i, j + k]) / (2.0 * k * DELTA))) + \
                      np.square((v[i, j + k] - v[i, j] + v[i + k, j + k] - v[i + k, j]) / (2.0 * k * DELTA)))
    return s

@nb.jit(nopython=True)
def edge(k, v) -> np.ndarray:
    k2 = int(k / 2)
    if k != 1:
        for i in range(0, N_X, k):
            for j in range(0, N_Y, k):
                v[i + k2, j + k2] = .25 * (v[i, j] + v[i + k, j] + v[i, j + k] + v[i + k, j + k])
                if i + k < N_X:
                    v[i + k, j + k2] = .5 * (v[i + k, j] + v[i + k, j + k])
                if j + k < N_Y:
                    v[i + k2, j + k] = .5 * (v[i, j + k] + v[i + k, j + k])
    return v

def check_TOL(sums: np.ndarray) -> np.bool8:
    return np.fabs((sums[-1] - sums[-2]) / sums[-2]) < TOL

def PoissonMultiGrid():
    v = np.zeros((N_X + 1, N_Y + 1))
    v = prepare_v(v)
    i = np.zeros(2)
    data_dict = {'i': [], 'sums': []}

    for k in K:
        data_dict['i'].append(f"#### {k} ####")
        data_dict['sums'].append(f"#### {k} ####")
        sums = [1]
        i[0] = i[1]
        while True:
            i[1] += 1
            v = it_1(k, v)
            sums.append(integral(k, v))
            if check_TOL(sums):
                break

            # iterations
            data_dict['i'].append(i[1])
            data_dict['sums'].append(sums[-1])

        v = edge(k, v)

        # heatmaps
        plt.title(f'k = {k}')
        plt.imshow(v.T, cmap='bwr')
        plt.imshow(np.flip(v[::k, ::k].T, 1), cmap='bwr')
        plt.colorbar()
        plt.savefig(f'V(x,y), k={k}.png')
        plt.clf()

    data = pd.DataFrame(data_dict)
    data.to_csv('S(it).csv', index=False)

def plotIntegral():
    s1 = pd.read_csv("Sk=16.csv")
    s2 = pd.read_csv("Sk=8.csv")
    s3 = pd.read_csv("Sk=4.csv")
    s4 = pd.read_csv("Sk=2.csv")
    s5 = pd.read_csv("Sk=1.csv")

    plt.title("S(it)")
    plt.plot(s1["i"], s1["sums"], "r", label="k=16")
    plt.plot(s2["i"], s2["sums"], "b", label="k=8")
    plt.plot(s3["i"], s3["sums"], "g", label="k=4")
    plt.plot(s4["i"], s4["sums"], "k", label="k=2")
    plt.plot(s5["i"], s5["sums"], "y", label="k=1")

    plt.savefig("Integral.png")

if __name__ == '__main__':
    PoissonMultiGrid()
    #plotIntegral()