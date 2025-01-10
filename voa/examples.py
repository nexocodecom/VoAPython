import numpy as np


def SR1(n):
    """
    SR1:
    X ~ Uniform(0, 1)
    Y = 2 + X + Normal(0, 1)
    """
    X = np.random.uniform(0, 1, n)
    Y = 2 + X + np.random.normal(loc=0, scale=1, size=n)
    return X, Y


def SR2(n):
    """
    SR2:
    X ~ Uniform(0, 1)
    Y = X^(0.25) + Normal(0, 0.25)
    """
    X = np.random.uniform(0, 1, n)
    Y = X**0.25 + np.random.normal(loc=0, scale=0.25, size=n)
    return X, Y


def SR3(n):
    """
    SR3:
    X ~ Uniform(0, 1)
    Y = Indicator(X >= 0.5) + Normal(0, 2)
    """
    X = np.random.uniform(0, 1, n)
    # (X >= 0.5) will be a boolean array; convert to int (0 or 1)
    Y = (X >= 0.5).astype(int) + np.random.normal(loc=0, scale=2, size=n)
    return X, Y


def SR4(n):
    """
    SR4:
    X ~ Normal(0, 1)
    Y = log(1 + |X|) + Normal(0, 1)
    """
    X = np.random.normal(loc=0, scale=1, size=n)
    Y = np.log(1 + np.abs(X)) + np.random.normal(loc=0, scale=1, size=n)
    return X, Y


def SR5(n):
    """
    SR5:
    X ~ Uniform(0, 1)
    Y = 4 * ((2*X - 1)^2 - 0.5)^2 + Normal(0, 0.5)
    """
    X = np.random.uniform(0, 1, n)
    Y = 4 * (((2 * X) - 1)**2 - 0.5)**2 + np.random.normal(loc=0, scale=0.5, size=n)
    return X, Y
