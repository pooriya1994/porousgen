import warnings
import numpy as np
import pytest

FLUID, SOLID = 0, 1


def n_fluid(g):
    return int(np.count_nonzero(g == FLUID))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield
