import argparse
import time
import math
import numpy as np
from scipy.linalg import null_space
import functions


def get_defined_functions(exe_num):
    if exe_num == 11.16:
        A = np.array([[1, 2, 1, 2], [1, 1, 2, 4]])
        c = np.array([1, 1.5, 1, 1])

        F = null_space(A)
        x0 = [[1], [0], [0], [1]]

        def f(x):
            z = x[0]
            t = x[1]
            f0 = np.dot((t * c), (np.dot(F, z) + x0))
            I_ = [math.log(np.dot(F[i, :], z) + x0[i]) for i in range(4)]

            # f = (t)*c'*(F*z + x0) -
            # (log(F(1,:)*z + x0(1))) - (log(F(2,:)*z + x0(2))) -
            # (log(F(3,:)*z + x0(3))) - (log(F(4,:)*z + x0(4)))

            return f0 + sum(I_)

        def g(x):
            z = x[0]
            t = x[1]
            logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
            logs_dev = np.array([[F[i, 0] / logs_divs[i] for i in range(4)],
                                 [F[i, 1] / logs_divs[i] for i in range(4)]])
            f0_dev = np.array([[F[i, 0] * t * np.conj(c[i]) for i in range(4)],
                               [F[i, 1] * t * np.conj(c[i]) for i in range(4)]])

            return sum(f0_dev) - sum(logs_dev)

        def H(x):
            z = x[0]
            logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
            logs_dev = np.array([[F[i, 0] / logs_divs[i] for i in range(4)],
                                 [F[i, 1] / logs_divs[i] for i in range(4)]])

            aux = logs_dev**2
            dz1z1 = aux[0]
            dz2z2 = aux[1]
            dz1z2 = np.array([F[i, 0] * F[i, 1] for i in range(4)]) * 1.0 / aux

            return np.array([[dz1z1, dz1z2], [dz1z2, dz2z2]])

        return f, g, H


def get_points(exe_num):
    if exe_num == 11.16:
        return [2, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    points = get_points(args.exe_num)

    if args.exe_num == 11.16:
        f, g, H = get_defined_functions(args.exe_num)
        t = time.time()
        x, fx, k = functions.iterative_conjugate_gradient(f, g, H, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(x, fx, k, delta_t))