import numpy as np


def integral(_func, start, end, n_split=500):
    interval = (end - start) / n_split
    return sum([_func(i)*interval for i in np.linspace(start, end, n_split)])


def comb(n, r):
    """The number of combinations of N things taken k at a time.
       This is known as "nCr".

    Parameters
    __________
    n: int
       Number of things.
    r: int
       Number of elements taken.

    Returns
    _______
    comb_n[r]: int
        The total number of combinations.

    Notes
    _____
    - "comb_n[i]" represents "nCi"
    - This algorithm is based on the formula,
        nCr = nCr-1 * (n - r + 1) / r

    """

    assert isinstance(n, int), "n and r should be natural numbers"
    assert isinstance(r, int), "n and r should be natural numbers"
    if n < r or n < 0:
        return 0

    comb_n = [0 for i in range(r+1)]
    comb_n[0] = 1
    for i in range(r):
        comb_n[i+1] = comb_n[i] * (n - i) / (i + 1)

    return comb_n[r]
