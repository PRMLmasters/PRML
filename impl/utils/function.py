import numpy as np
import functools

def integral(_func, start, end, n_split=500, args=None):
    interval = (end - start) / n_split
    args = () if args is None else args
    return sum([_func(i, *args) * interval for i in np.linspace(
                start, end, n_split)])


# 範囲の引数が無限の時に,有限の値になるように制限
def _limit_inf(n_bound, limit=1000):
    if not np.isinf(num):
        return num
    else:
        return limit if num > 0 else -limit


# ほぼscipyの実装通り
def n_dimension_integral(_func, ranges, n_split=500, args=None):
    """
    Parameters
    __________
    ranges : ([loe, high],[loe, high],...)
    _func : function
    n_split : int number of splits of range
    """
    ranges = [_RangeProcessing(_range) for _range in ranges]
    depth = len(ranges)
    args = () if args is None else args
    return _NQuad(_func, ranges).integral(*args)


class _NQuad:

    def __init__(self, _func, _ranges, _n_split=500):
        self.func = _func
        self.ranges = _ranges
        self.max_depth = len(_ranges)
        self.n_split = _n_split

    def integral(self, *args, **kwargs):
        depth = kwargs.pop("depth", 0)
        _range = self.ranges[-1-depth]
        low, high = _range()
        if depth < self.max_depth - 1:
            f = functools.partial(self.integral, depth=depth+1)
        else:
            f = self.func
        quad = integral(f, low, high, n_split=self.n_split, args=args)
        return quad


class _RangeProcessing:

    def __init__(self, _range):
        self.range = _range

    def __call__(self):
        return self.range


def random_sampled_n_intergral(_func, ranges, n_sample=10000, args=None):
    """
    ranges...([low,high],[low,high],...)
    """
    ranges = tuple(list(map(limit_inf, _range)) for _range in ranges)
    interval = np.prod([(_range[1] - _range[0]) for _range in ranges],
                       axis=0) / num_sample
    combinations = np.asarray([
        np.random.uniform(*_range, num_sample) for _range in ranges]).T
    args = () if args is None else args
    return sum([_func(*combi, *args) * interval for combi in combinations])
