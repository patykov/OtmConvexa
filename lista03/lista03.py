import argparse
import time

import numpy as np

import functions


def get_defined_functions(exe_num):
    if exe_num == 6.1:
        Q1 = [[12, 8, 7, 6], [8, 12, 8, 7], [7, 8, 12, 8], [6, 7, 8, 12]]
        Q2 = [[3, 2, 1, 0], [2, 3, 2, 1], [1, 2, 3, 2], [0, 1, 2, 3]]
        Q3 = [[2, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]]
        Q4 = np.eye(4)

        Q = np.concatenate((np.concatenate(
            (Q1, Q2, Q3, Q4), axis=1), np.concatenate(
                (Q2, Q1, Q2, Q3), axis=1), np.concatenate(
                    (Q3, Q2, Q1, Q2), axis=1), np.concatenate((Q4, Q3, Q2, Q1), axis=1)),
                           axis=0)
        b = [-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0]

        def f(x):
            return 0.5 * np.dot(np.dot(x, Q), x) + np.dot(b, x)

        def g(x):
            return b + np.dot(Q, x)

        def H(x):
            return Q

        return f, g, H

    elif exe_num == 6.2:

        def f(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

        def g(x):
            a = 200 * (x[1] - x[0]**2)
            return np.array([(-2 * x[0] * a) - (2 * (1 - x[0])), a])
            # [2 * x[0] - 400 * x[0] * (-x[0]**2 + x[1]) - 2, -200 * x[0]**2 + 200 * x[1]]

        def get_f(x0, d):
            def f(alpha):
                x = x0 + alpha * d
                return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

            return f

        def get_f_(x0, d):
            def f_(alpha):
                x = x0 + alpha * d
                return 200 * (x[1] - x[0]**2) * (d[1] - 2 * x[0] * d[0]) - 2 * d[0] * (1 - x[0])

            return f_

        return f, g, get_f, get_f_

    elif exe_num == 6.3:

        def f(x):
            return 5 * x[0]**2 - 9 * x[0] * x[1] + 4.075 * x[1]**2 + x[0]

        def g(x):
            return np.array([10 * x[0] - 9 * x[1] + 1, 8.15 * x[1] - 9 * x[0]])

        def H(x):
            return np.array([[10, -9], [-9, 8.15]])

        return f, g, H


def get_points(exe_num):
    if exe_num == 6.1:
        return [1] * 16

    if exe_num == 6.2:
        return [[-2, 2], [2, -2], [-2, -2]]

    if exe_num == 6.3:
        return [1, 1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    # parser.add_argument('-min', help='Initial min value.', required=True, type=float)
    # parser.add_argument('-max', help='Initial max value.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    points = get_points(args.exe_num)

    # functions.plot3d_f(f, args.exe_num, args.min, args.max)

    if args.exe_num == 6.1:  # fazer para mais pontos
        f, g, H = get_defined_functions(args.exe_num)
        t = time.time()
        x, fx, k = functions.gradient_descent(f, g, H, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(x, fx, k, delta_t))

    elif args.exe_num == 6.2:
        f, g, get_f, get_f_ = get_defined_functions(args.exe_num)
        for p in points:
            print('Ponto Inicial: {}'.format(p))
            t = time.time()
            x, fx, k = functions.fletcher_reeves(f, g, get_f, get_f_, p, eps)
            delta_t = (time.time() - t) * 1000
            print('x: {}, fx: {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(x, fx, k, delta_t))

    elif args.exe_num == 6.3:  # fazer para mais pontos
        f, g, H = get_defined_functions(args.exe_num)
        t = time.time()
        x, fx, k = functions.gradient_descent(f, g, H, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(x, fx, k, delta_t))
