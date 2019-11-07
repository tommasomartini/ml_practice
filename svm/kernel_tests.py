import unittest

import numpy as np

import svm.kernels as kernels


class KernelTests(unittest.TestCase):

    def test_linear_kernel(self):
        v1 = np.array([1, 2, 0])
        v2 = np.array([[-1, -1, 10],
                       [3, 4, 10]])

        expected_output = np.array([-2, 12])

        linear_kernel = kernels.get_linear_kernel()
        output = linear_kernel(v1, v2)

        np.testing.assert_array_almost_equal(expected_output, output)


if __name__ == '__main__':
    unittest.main()
