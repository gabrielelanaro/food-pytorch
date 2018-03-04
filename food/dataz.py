import dask
from dask.base import DaskMethodsMixin
from dask.optimization import cull
from toolz import groupby, pluck, concat
from itertools import combinations, combinations_with_replacement

def lpluck(key, iterable):
    return list(pluck(key, iterable))

def example():

    collection = Collection.from_list([('cheese', 1),
                                       ('cheese', 2),
                                       ('cheese', 3),
                                       ('tiramisu', 10),
                                       ('tiramisu', 11),
                                       ('meat', 12)])

    collection = (collection
                  .groupby(0)
                  .starmap(lambda key, val: (key, lpluck(1, val))))

    stuff = (collection
             .combinations(replacement=True)
             .starmap(generate_matching_pairs)
             .flatten())
    print(stuff.compute())


def generate_matching_pairs(group1, group2):
    same_key = group1[0] == group2[0]
    return [(same_key, k1, k2) for k1 in group1[1] for k2 in group2[1]]

class Collection(DaskMethodsMixin):
    def __init__(self, dsk, keys):
        self._dsk = dsk
        self._keys = list(keys)

    @classmethod
    def from_list(cls, data):
        dsk = {f'collection-{i}': d for i, d in enumerate(data)}
        return cls(dsk, dsk.keys())

    def map(self, function):
        dsk = self._dsk.copy()

        new_call = [(k + '-' + function.__name__, (function, k)) for k in  self._keys]
        dsk.update(new_call)
        return Collection(dsk, list(pluck(0, new_call)))

    def starmap(self, function):
        return self.map(lambda a: function(*a))

    def groupby(self, key):
        results = self.compute()
        return Collection.from_list(groupby(key, results).items())

    def product(self, other):
        dsk = self._dsk.copy()
        dsk.update({'other-' + k: v for k, v in other._dsk.items()})

        new_dsk = [('prod-' + k1 + k2, (k1, k2))
                   for k1 in self._keys for k2 in prepend('other-', other._keys)]
        dsk.update(new_dsk)
        return Collection(dsk, pluck(0, new_dsk))

    def combinations(self, replacement=False):
        if replacement:
            func = combinations_with_replacement
        else:
            func = combinations
        dsk = self._dsk.copy()
        new_dsk = [(f'comb-{i}', (tuple, list(v))) for i, v in enumerate(func(self._keys, 2))]
        dsk.update(new_dsk)
        return Collection(dsk, lpluck(0, new_dsk))

    def flatten(self):
        # This will trigger computation
        return Collection.from_list(concat(self.compute()))

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, key):
        return self.__dask_scheduler__(self._dsk, self._keys[key])

    def __dask_graph__(self):
        return self._dsk

    def __dask_keys__(self):
        return self._keys

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
        return list, ()

    def __dask_postpersist__(self):
        # Since our __init__ takes a graph as its first argument, our
        # rebuild function can just be the class itself. For extra
        # arguments we also return a tuple containing just the keys.
        return Collection, (self._keys,)

    def __dask_tokenize__(self):
        # For tokenize to work we want to return a value that fully
        # represents this object. In this case it's the list of keys
        # to be computed.
        return tuple(self._keys)

def prepend(pref, iterable):
    return [pref + i for i in iterable]
