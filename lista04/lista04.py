import argparse
import time
import math
import numpy as np
from scipy.linalg import null_space
import functions


def restricted_log(x):
    if x >= 0.0:
        return -np.inf
    else:
        return math.log(-x)


def get_defined_functions(exe_num):
    if exe_num == 11.16:
        A = np.array([[1, 2, 1, 2], [1, 1, 2, 4]])
        b = np.array((3, 5))
        c = np.array([1, 1.5, 1, 1])

        F = null_space(A)

        # Getting initial x
        x0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        def f(x):
            z = x[0]
            t = x[1]
            f0 = np.dot((t * c), (np.dot(F, z) + x0))
            I_ = [restricted_log(np.dot(F[i, :], z) + x0[i]) for i in range(4)]

            return f0 - sum(I_)

        def g(x):
            z = x[0]
            t = x[1]
            logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
            logs_dev = np.array([
                sum([F[i, 0] / logs_divs[i] for i in range(4)]),
                sum([F[i, 1] / logs_divs[i] for i in range(4)])
            ])
            f0_dev = np.array([
                sum([F[i, 0] * t * np.conj(c[i]) for i in range(4)]),
                sum([F[i, 1] * t * np.conj(c[i]) for i in range(4)])
            ])

            return f0_dev - logs_dev

        def H(x):
            z = x[0]
            logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
            logs_dev = np.array([[F[i, 0] / logs_divs[i] for i in range(4)],
                                 [F[i, 1] / logs_divs[i] for i in range(4)]])
            aux = logs_dev**2
            dz1z1 = sum(aux[0])
            dz2z2 = sum(aux[1])
            dz1z2 = sum([(F[i, 0] * F[i, 1]) / logs_divs[i]**2 for i in range(4)])

            return np.array([[dz1z1, dz1z2], [dz1z2, dz2z2]])

        def original(z):
            min_x = np.dot(F, z) + x0
            min_fx = np.dot(c, min_x)

            return min_x, min_fx

        return x0, f, g, H, original


def get_points(exe_num):
    if exe_num == 11.16:
        return [0, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    points = get_points(args.exe_num)

    if args.exe_num == 11.16:
        x0, f, g, H, original = get_defined_functions(args.exe_num)
        print('Initial point: {}\n'.format(x0))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(f, g, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(f, g, H, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        # print('Quasi-Newton')
        # t = time.time()
        # x, fx, k, it = functions.iterative_conjugate_gradient(f, g, H, original, points, eps)
        # delta_t = (time.time() - t) * 1000
        # print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
        #     x, fx, k, it, delta_t))
