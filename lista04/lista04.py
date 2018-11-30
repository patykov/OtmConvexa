import argparse
import time
import numpy as np
from scipy.linalg import null_space
import functions


def get_defined_functions(exe_num):
    if exe_num == 11.16:
        A = np.array([[1, 2, 1, 2], [1, 1, 2, 4]])
        b = np.array([[3], [5]])
        c = np.array([1, 1.5, 1, 1])

        F = null_space(A)
        x0 = [[1], [0], [0], [1]]

        def f(z, t):
            f0 = np.dot((t * c), (np.dot(F, z) + x0))
            I_ = [np.ln(np.dot(F[i, :], z) + x0[i]) for i in range(4)]

            # f = (t)*c'*(F*z + x0) -
            # (log(F(1,:)*z + x0(1))) - (log(F(2,:)*z + x0(2))) - (log(F(3,:)*z + x0(3))) - (log(F(4,:)*z + x0(4)))

            return f0 + sum(I_)

        def g(z, t):
            divs = [x0[i] + F[i,1]*z[1] + F[i,2]*z[2]]
            g = [
                F11*t*conj(c1) - F21/(x02 + F21*z1 + F22*z2) - F31/(x03 + F31*z1 + F32*z2) - F41/(x04 + F41*z1 + F42*z2) - F11/(x01 + F11*z1 + F12*z2) + F21*t*conj(c2) + F31*t*conj(c3) + F41*t*conj(c4);
                F12*t*conj(c1) - F22/(x02 + F21*z1 + F22*z2) - F32/(x03 + F31*z1 + F32*z2) - F42/(x04 + F41*z1 + F42*z2) - F12/(x01 + F11*z1 + F12*z2) + F22*t*conj(c2) + F32*t*conj(c3) + F42*t*conj(c4)];


        return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    # points = get_points(args.exe_num)

    if args.exe_num == 11.16:
        f, g, H = get_defined_functions(args.exe_num)
        t = time.time()
        x, fx, k = functions.iterative_conjugate_gradient(f, g, H, p, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(x, fx, k, delta_t))