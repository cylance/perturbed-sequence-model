import numpy as np
import pytest


def get_argmax_of_matrix_as_tuple(mat):
    return np.unravel_index(mat.argmax(), mat.shape)
