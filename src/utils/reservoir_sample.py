from math import exp, log, floor
from random import random, randrange
from itertools import islice
from io import StringIO
import pandas as pd


def reservoir_sample(iterable, k=1):
    """Select k items uniformly from iterable.

    Returns the whole population if there are k or fewer items

    from https://bugs.python.org/issue41311#msg373733
    """
    iterator = iter(iterable)
    values = list(islice(iterator, k))

    W = exp(log(random())/k)
    while True:
        # skip is geometrically distributed
        skip = floor(log(random())/log(1-W))
        selection = list(islice(iterator, skip, skip+1))
        if selection:
            values[randrange(k)] = selection[0]
            W *= exp(log(random())/k)
        else:
            return values


def sample_file(filepath, k):
    with open(filepath, 'r') as f:
        header = next(f)
        result = [header] + reservoir_sample(f, k)
        df = pd.read_csv(StringIO(''.join(result)))
    return df
