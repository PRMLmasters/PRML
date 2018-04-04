import numpy as np


def integral(_func, start, end, n_split=500):
    interval = (end - start) / n_split
    return sum([_func(i)*interval for i in np.linspace(start, end, n_split)])


def comb(n, r):
    assert isinstance(n, 1) and isinstance(r, 1),
    "n and r should be natural numbers"
    if n < r or n < 0:
        return 0

    comb_n = [0 for i in range(r+1)]
    comb_n[0] = １
    for i in range(r):
        comb_n[i+1] = comb_n[i] * (n - i) / (i + 1)

    return comb_n[r]
