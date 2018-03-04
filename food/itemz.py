import dask
from dask.utils import funcname
from itertools import combinations, combinations_with_replacement, starmap
from toolz.curried import groupby, map, pluck, curry, concat, filter
from toolz import merge
from dask.base import DaskMethodsMixin
from dask.optimization import cull
import random
from operator import itemgetter

def example():
    data = [('cheese', 1),
       ('cheese', 2),
       ('cheese', 3),
       ('tiramisu', 10),
       ('tiramisu', 11),
       ('meat', 12)]

    c = Collection.create(data)
    c_train, c_test = c.random_split(0.5, seed=1)

    pipe = (Collection.pipeline("file_list")
                      .groupby(0)
                      .evolve(1, lambda x: list(pluck(1, x)))
                      .combinations(replacement=True)
                      .starmap(generate_matching_pairs)
                      .flatten()
                      .filter(random_drop(0.5)))

    print(pipe.pump({'file_list': c_train}).compute())
    print(pipe.pump({'file_list': c_test}).compute())

def generate_matching_pairs(group1, group2):
    same_key = group1[0] == group2[0]
    return [(same_key, k1, k2) for k1 in group1[1] for k2 in group2[1]]

@curry
def random_drop(proportion, seed, data):
    random.seed(seed)
    if random.random() < proportion and data[0] is False:
        return False
    else:
        return True

def _error(name):
    raise ValueError(f'Entry point {name} is required')

def _identity(x):
    return x

class Cell(DaskMethodsMixin):

    def __init__(self, dsk, key):
        self._dsk = dsk
        self._key = key

    @classmethod
    def create(cls, val, name='input'):
        return cls({name: val}, name)

    @classmethod
    def pipeline(cls, entrypoint):
        return cls.create((_error, entrypoint), entrypoint)

    def pump(self, data):
        cls = type(self)
        dsk = self._dsk.copy()
        deps = merge(d._dsk for d in data.values() if isinstance(d, Cell))
        data = {k: (_identity, d._key) if isinstance(d, Cell) else d for k, d in data.items()}
        dsk.update(deps)
        dsk.update(data)
        return cls(dsk, self._key)

    def apply(self, function):
        cls = type(self)
        dsk = self._dsk
        node = self._key + '-apply-' + funcname(function)
        dsk.update({node: (function, self._key)})
        return cls(dsk, node)

    def __dask_graph__(self):
        return self._dsk

    def __dask_keys__(self):
        return self._key

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        dsk2, _ = cull(dsk, keys)
        return dsk2

    # Use the threaded scheduler by default.
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def __dask_postcompute__(self):
        # We want to return the results as a tuple, so our finalize
        # function is `tuple`. There are no extra arguments, so we also
        # return an empty tuple.
        return lambda x: x, ()

    def __dask_postpersist__(self):
        # Since our __init__ takes a graph as its first argument, our
        # rebuild function can just be the class itself. For extra
        # arguments we also return a tuple containing just the keys.
        return type(self), (self._key,)

    def __dask_tokenize__(self):
        # For tokenize to work we want to return a value that fully
        # represents this object. In this case it's the list of keys
        # to be computed.
        return self._key


class Collection(Cell):

    def groupby(self, key):
        return self.apply(groupby(0)).apply(dict.items)

    def map(self, func):
        return self.apply(map(func))

    def filter(self, func):
        return self.apply(filter(func))

    def starmap(self, func):
        return self.apply(lambda x: starmap(func, x))

    def evolve(self, key, func):
        return self.map(_evolve(key, func))

    def combinations(self, k=2, replacement=False):
        if replacement:
            func = combinations_with_replacement
        else:
            func = combinations
        return self.apply(lambda x: func(x, k))

    def flatten(self):
        # This will trigger computation
        return self.apply(concat)

    def take(self, index):
        return self.apply(itemgetter(index))

    def random_split(self, proportion, seed=None):
        splits = self.apply(_random_split(proportion, seed))
        return splits.split(2)

    def split(self, n):
        return [self.take(i) for i in range(n)]

    def __dask_postcompute__(self):
        # We want to return the results as a tuple, so our finalize
        # function is `tuple`. There are no extra arguments, so we also
        # return an empty tuple.
        return list, ()

@curry
def _evolve(key, func, data):
    if isinstance(data, list):
        data = data[:] # make a copy
        data[key] = func(data[key])
        return data
    elif isinstance(data, tuple):
        data = list(data)
        data[key] = func(data[key])
        return tuple(data)
    elif isinstance(data, dict):
        data = data.copy()
        data[key] = func(data[key])
        return data
    else:
        raise ValueError("Accepted types for evolve are list, tuple or dict")

from collections import Iterator

@curry
def _random_split(proportion, seed, data):
    if not (0 < proportion < 1):
        raise ValueError('Proportion should be between 0 and 1')

    if seed:
        random.seed(seed)

    split1 = []
    split2 = []
    if isinstance(data, Iterator):
        data = list(data)

    if isinstance(data, list):
        data = data[:] # make a copy
        random.shuffle(data)
        offset = int(proportion*len(data))
        return data[:offset], data[offset:]
    else:
        raise ValueError(f"Accepted types for random split are list, received {type(data)}")
