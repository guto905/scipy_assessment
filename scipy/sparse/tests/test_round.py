import unittest
import numpy as np
from scipy.sparse import csr_matrix

class TestCSRMatrixRound(unittest.TestCase):
    def test_round_sparse_matrix(self):
        data = np.array([1.111, 2.222, 3.333])
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 3])
        mat = csr_matrix((data, indices, indptr), shape=(1, 4))

        rounded = round(mat, 2)

        expected = csr_matrix(([1.11, 2.22, 3.33], indices, indptr), shape=(1, 4))
        np.testing.assert_array_almost_equal(rounded.data, expected.data, decimal=2)
