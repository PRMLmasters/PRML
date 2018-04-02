import numpy as np


def integral(_func, start, end, n_split=500):
    interval = (end - start) / n_split
    return sum([_func(i)*interval for i in np.linspace(start, end, n_split)])
