import numpy as np
import numba
import time
import plotly.graph_objs as go
from typing import List, Dict, Tuple, Any
import math

@numba.njit(parallel=True)
def Q_function(Q_grid: np.ndarray, C_grid: np.ndarray) -> Dict[str, np.ndarray]:
    n = C_grid.shape[0] - 1  # Because C_grid is (n+1) x (n+1)
    M = Q_grid.shape[0]

    x = np.empty(M, dtype=np.float64)
    y = np.empty(M, dtype=np.float64)
    z = np.empty(M, dtype=np.float64)

    for i in numba.prange(M):
        xi = Q_grid[i, 0]
        yi = Q_grid[i, 1]
        x[i] = xi
        y[i] = yi

        # Compute row/col indices
        row_idx = int(math.ceil(xi * (n + 1)) - 1)
        col_idx = int(math.ceil(yi * (n + 1)) - 1)

        # Clamp indices to [0, n]
        if row_idx > n:
            row_idx = n
        if col_idx > n:
            col_idx = n

        # Lookup value in C_grid
        z[i] = C_grid[row_idx, col_idx]

    return {"x": x, "y": y, "z": z}


def create_Q_grid(n: int = 10) -> np.ndarray:
    u = np.linspace(0.5 / (n + 1), (0.5 + n) / (n + 1), n + 1)
    grid = np.stack(np.meshgrid(u, u)).T.reshape(-1, 2)

    return grid


@numba.njit(parallel=True)
def calculate_copula_part(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = len(t)
    n_inv = 1.0 / float(n)
    for i in numba.prange(1, int(n / 2) + 2):
        for j in numba.prange(int(n / 2) + 2):
            c[i, j] = c[0, j] + n_inv * np.sum(j >= t[:i])
    return c


@numba.njit
def wn(n: int, c: float, i: int, j: int) -> float:
    u = (i + 0.5) / (n + 1)
    v = (j + 0.5) / (n + 1)
    return n**0.5 * (c - u * v) * (u * v * (1 - u) * (1 - v)) ** -0.5


@numba.njit
def fill_matrix(n: int, ks: np.ndarray, c: np.ndarray, x_prim: bool, y_prim: bool) -> np.ndarray:
    sign = (-1) ** (x_prim + y_prim)

    for i in range(int(np.floor((n + 1) / 2)) + 1):
        for j in range(int(np.floor((n + 1) / 2)) + 1):
            x = n - i if x_prim else i
            y = n - j if y_prim else j
            ks[x, y] = sign * wn(n, c[i, j], i, j)
    return ks


@numba.njit
def arma_copula(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    n = len(rx)

    ctab = np.zeros((n + 1, n + 1))
    ctabs22 = np.zeros((n + 1, n + 1))
    ctabs12 = np.zeros((n + 1, n + 1))
    ctabs21 = np.zeros((n + 1, n + 1))

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
def calculate_copula_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return arma_copula(
        np.argsort(np.argsort(x)) + 1, np.argsort(np.argsort(y)) + 1
    )


def calculate_copula_mc_grid(x: np.ndarray, y: np.ndarray, mc: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    k = len(x)
    g = np.zeros((k + 1, len(y) + 1))

    for _ in range(mc):
        indx = rng.choice(k, size=k)
        mat = calculate_copula_grid(x[indx], y[indx])
        g += mat / mc

    return g


def create_Q_plot(X: List[float], Y: List[float], k_plot_grid: int = 100, MC: int = 100, display: bool = True) -> Dict:
    if len(X) != len(Y):
        raise ValueError("Size of X and Y do not match")

    # Start the timer
    start_time = time.time()

    # Calculate the copula grid using Monte Carlo estimation
    C_grid = calculate_copula_mc_grid(np.array(X), np.array(Y), MC)

    # Create the Q grid
    Q_grid = create_Q_grid(k_plot_grid)

    # Calculate the Q function on the grid
    plot_points = Q_function(Q_grid, C_grid)

    # Stop the timer
    time_taken = time.time() - start_time
    print(f"Time taken for calculations: {time_taken:.2f} seconds")

    # Create a contour plot using Plotly
    contour_plot = go.Contour(
        z=plot_points['z'],
        x=plot_points['x'],
        y=plot_points['y'],
        contours=dict(
            coloring='jet'
        )
    )

    layout = go.Layout(
        title='Q Function Contour Plot',
        xaxis_title='X',
        yaxis_title='Y',
        height=600,
        width=600
    )

    fig = go.Figure(data=[contour_plot], layout=layout)

    # Display the plot if the display flag is True
    if display:
        fig.show()

    # Return the plot and data as a dictionary
    return {
        'Q_plot': fig,
        'C_grid': C_grid,
        'Q_grid': Q_grid,
        'plot_points': plot_points
    }