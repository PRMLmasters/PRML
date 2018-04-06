import numpy as np


def integral(_func, start, end, n_split=500, args=None):
    interval = (end - start) / n_split
    return sum([_func(i)*interval for i in np.linspace(
        start, end, n_split)]) if args is None else \
            sum([_func(i,*args)*interval for i in np.linspace(
                start, end, n_split)])

#範囲の引数が無限の時に,有限の値になるように制限
def limit_inf(num, limit=1000):
    if not np.isinf(num):
        return num
    else:
        return limit if num > 0 else -limit
    
def n_dimension_integral(_func, ranges, n_split=500, args=None):
    """
    ranges...([low,high],[low,high],...)
    """
    ranges = tuple(list(map(limit_inf ,_range)) for _range in ranges)
    interval = np.prod([(_range[1] - _range[0])/ n_split for _range in ranges], axis=0)
    ranges = tuple(list(map(lambda x: np.linspace(*x, n_split), ranges)))
    combinations = np.asarray([_range.ravel() for _range in np.meshgrid(*ranges)]).T
    return sum([_func(*combi,*args)*interval for combi in combinations]) \
        if args is not None else \
            sum([_func(*combi)*interval for combi in combinations])

def random_sampled_n_intergral(_func, ranges, num_sample=10000, args=None):
    """
    ranges...([low,high],[low,high],...)
    """
    ranges = tuple(list(map(limit_inf ,_range)) for _range in ranges)
    interval = np.prod([(_range[1] - _range[0]) for _range in ranges], axis=0) / num_sample
    combinations = np.asarray([
        np.random.uniform(*_range,num_sample) for _range in ranges]).T
    print(interval,combinations.shape)
    return sum([_func(*combi,*args)*interval for combi in combinations]) \
        if args is not None else \
            sum([_func(*combi)*interval for combi in combinations])
