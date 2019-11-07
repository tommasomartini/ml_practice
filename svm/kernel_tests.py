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

    def test_polynomial_kernel_p_2(self):
        v1 = np.array([1, 2, 0])
        v2 = np.array([[-1, -1, 10],
                       [3, 4, 10]])

        expected_output = np.array([4, 144])

        polynomial_kernel = kernels.get_polynomial_kernel(p=2)
        output = polynomial_kernel(v1, v2)

        np.testing.assert_array_almost_equal(expected_output, output)

    def test_polynomial_kernel_p_3(self):
        v1 = np.array([1, 2, 0])
        v2 = np.array([[-1, -1, 10],
                       [3, 4, 10]])

        expected_output = np.array([-8, 12 ** 3])

        polynomial_kernel = kernels.get_polynomial_kernel(p=3)
        output = polynomial_kernel(v1, v2)

        np.testing.assert_array_almost_equal(expected_output, output)

    def test_radial_basis_function_kernel(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([[1, 1, -1],
                       [-2, 3, 5]])

        expected_output = np.array([0.15123976, 0.211072088])

        polynomial_kernel = kernels.get_radial_basis_function_kernel(sigma=3)
        output = polynomial_kernel(v1, v2)

        np.testing.assert_array_almost_equal(expected_output, output)


if __name__ == '__main__':
    unittest.main()
