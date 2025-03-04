import collections
import itertools
import numpy as np
import torch
import torch.nn as nn


# Source: webdataset.utils
def make_seed(*args):
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


def gather(x: torch.Tensor, index: torch.Tensor, dim: int, dims: tuple=None) -> torch.Tensor:
    if dims is None:
        dims = tuple(range(len(index.shape)))

    other_dims = list(range(len(x.shape)))
    for d in dims[::-1]:
        other_dims.pop(d)

    expand_vals = len(dims) * [-1]
    for d in other_dims:
        index = index.unsqueeze(d)
        expand_vals.insert(d, x.shape[d])

    index = index.expand(*expand_vals)

    x = torch.gather(x, dim=dim, index=index)

    return x



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)


def basic_module_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
