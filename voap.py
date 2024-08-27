import numpy as np
import numba
# from tqdm.auto import tqdm


@numba.njit(parallel=True)
def Q_function(Q_grid, C_grid):
    n = len(C_grid) - 1
    z = []

    for i in numba.prange(len(Q_grid)):
        row_idx = int(np.ceil(Q_grid[i, 0] * (n + 1)) - 1)
        col_idx = int(np.ceil(Q_grid[i, 1] * (n + 1)) - 1)
        if row_idx > n:
            row_idx = n
        if col_idx > n:
            col_idx = n
        z.append(C_grid[row_idx, col_idx])

    return {"x": Q_grid[:, 0], "y": Q_grid[:, 1], "z": np.array(z)}


def create_Q_grid(n=10):
    u = np.linspace(0.5 / (n + 1), (0.5 + n) / (n + 1), n + 1)
    grid = np.stack(np.meshgrid(u, u)).T.reshape(-1, 2)

    return grid

@numba.njit(parallel=True)
def calculate_copula_part(t, c):
    n = len(t)
    n_inv = 1 / n # n**-1 ## numba doesn't get n**-1 xD
    c = np.zeros(c.shape)
    for i in numba.prange(1, int(n / 2) + 2):
        for j in numba.prange(int(n / 2) + 2):
            # c[i, j] = c[i - 1, j] + (j >= t[i - 1]) * n_inv
            c[i, j] = c[0, j] + n_inv * np.sum(j >= t[:i])
    return c

@numba.njit
def wn(n, c, i, j):
    u = (i + 0.5) / (n + 1)
    v = (j + 0.5) / (n + 1)
    return n**0.5 * (c - u * v) * (u * v * (1 - u) * (1 - v)) ** -0.5

@numba.njit
def fill_matrix(n, ks, c, x_prim, y_prim):
    sign = (-1) ** (x_prim + y_prim)

    for i in range(int(np.floor((n + 1) / 2)) + 1):
        for j in range(int(np.floor((n + 1) / 2)) + 1):
            x = n - i if x_prim else i
            y = n - j if y_prim else j
            ks[x, y] = sign * wn(n, c[i, j], i, j)
    return ks

@numba.njit
def arma_copula(rx, ry):
    n = len(rx)

    ctab = np.zeros((n + 1, n + 1))
    ctabs22 = np.zeros((n + 1, n + 1))
    ctabs12 = np.zeros((n + 1, n + 1))
    ctabs21 = np.zeros((n + 1, n + 1))

    rsx = np.zeros(n)
    rsy = np.zeros(n)

    ks = np.zeros((n + 1, n + 1))

    rsx = len(rx) + 1 - rx
    rsy = len(ry) + 1 - ry

    t = ry[np.argsort(rx)]
    ts22 = rsy[np.argsort(rsx)]
    ts12 = rsy[np.argsort(rx)]
    ts21 = ry[np.argsort(rsx)]

    ctab = calculate_copula_part(t, ctab)
    ctabs22 = calculate_copula_part(ts22, ctabs22)
    ctabs12 = calculate_copula_part(ts12, ctabs12)
    ctabs21 = calculate_copula_part(ts21, ctabs21)

    ks = fill_matrix(n, ks, ctab, False, False)
    ks = fill_matrix(n, ks, ctabs22, True, True)
    ks = fill_matrix(n, ks, ctabs12, False, True)
    ks = fill_matrix(n, ks, ctabs21, True, False)

    return ks

@numba.njit
def calculate_copula_grid(x, y):
    return arma_copula(
        np.argsort(np.argsort(x)) + 1, np.argsort(np.argsort(y)) + 1
    )

# @numba.njit(parallel=True)
def calculate_copula_mc_grid(x, y, mc=100, seed=0):
    rng = np.random.RandomState(seed)
    k = len(x)
    g = np.zeros((k + 1, len(y) + 1))

    for i in range(mc):
        indx = rng.choice(k, size=k)
        mat = calculate_copula_grid(x[indx], y[indx])
        g += mat / mc

    return g


# def create_Q_plot(x, y):
#     return Q_function(create_Q_grid(), calculate_copula_mc_grid(x, y))
