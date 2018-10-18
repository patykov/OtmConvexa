import argparse
import time
import math
import numpy as np

import functions


def get_f(num):
    if num == 4.2:

        def f(x):
            return -5 * x**5 + 4 * x**4 - 12 * x**3 + 11 * x**2 - 2 * x + 1

    elif num == 4.3:

        def f(x):
            return math.log(x - 2)**2 + math.log(10 - x)**2 - x**(0.2)

    elif num == 4.4:

        def f(x):
            return -3 * x * math.sin(0.75 * x) + math.e**(-2 * x)

    return f


def get_f_(num):
    if num == 4.2:

        def f_(x):
            return -25 * x**4 + 16 * x**3 - 36 * x**2 + 22 * x - 2

    elif num == 4.3:

        def f_(x):
            return 2 * (math.log(x - 2) / (x - 2) + math.log(10 - x) / (x - 10)) - 0.2 * x**(-0.8)

    elif num == 4.4:

        def f_(x):
            return -3 * math.sin(0.75 * x) - 2.25 * x * math.cos(0.75 * x) - 2 * math.e**(-2 * x)

    return f_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    parser.add_argument('-min', help='Initial min value.', required=True, type=float)
    parser.add_argument('-max', help='Initial max value.', required=True, type=float)
    parser.add_argument('-eps', help='Initial max value.', required=False, default=10**(-5))
    args = parser.parse_args()

    f = get_f(args.exe_num)
    f_ = get_f_(args.exe_num)

    if args.exe_num == 4.2:
        cubic_values = [[-0.2, 0.1, 0.2], [-0.3, 0.1, 0.3], [-0.5, 0.1, 0.5]]
        davies_values = [0.1, 0.2, 0.5]  # Dont work with negative or zero
        backtrack_values = [-0.5, 0.2, 0.5]

    if args.exe_num == 4.3:
        cubic_values = [[7, 8, 9], [6.9, 7, 8], [6, 8, 9], [8, 8.5, 9]]
        davies_values = [7, 8.5, 9]  # Dont work with negative or zero
        backtrack_values = [6, 7, 8, 9]

    if args.exe_num == 4.4:
        cubic_values = [[0, np.pi, 2 * np.pi], [0, 1.0, np.pi], [1, 2, 3], [2, 3, 4]]
        davies_values = [0, 2, np.pi, 2 * np.pi]  # Dont work with negative or zero
        backtrack_values = [0, 2, np.pi, 2 * np.pi]

    print('Exercise {}:\n'.format(args.exe_num))
    t = time.time()
    min_x, min_f_x, num_iter = functions.dichotomos_search(f, args.min, args.max, args.eps)
    delta_t = (time.time() - t) * 1000
    print('Dichotomous Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    min_x, min_f_x, num_iter = functions.fibonacci_search(f, args.min, args.max, args.eps)
    delta_t = (time.time() - t) * 1000
    print('\nFibonacci Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    min_x, min_f_x, num_iter = functions.golden_section_search(f, args.min, args.max, args.eps)
    delta_t = (time.time() - t) * 1000
    print('\nGolden Section Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    min_x, min_f_x, num_iter = functions.quadratic_interpolation_search(
        f, args.min, args.max, args.eps)
    delta_t = (time.time() - t) * 1000
    print('\nQuadratic Interpolation Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms'.format(
        min_x, min_f_x, num_iter, delta_t))

    print('\nCubic Interpolation Search')
    for values in cubic_values:
        t = time.time()
        min_x, min_f_x, num_iter = functions.cubic_interpolation_search(
            f, f_, values[0], values[1], values[2], args.eps)
        delta_t = (time.time() - t) * 1000
        print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms, for {}'.format(
            min_x, min_f_x, num_iter, delta_t, values))

    print('\nDavies, Swann and Campey Search')
    for value in davies_values:
        t = time.time()
        min_x, min_f_x, num_iter = functions.davies_swann_campey(f, value, args.eps)
        delta_t = (time.time() - t) * 1000
        print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms, for {}'.format(
            min_x, min_f_x, num_iter, delta_t, value))

    print('\nBacktraking Line Search')
    for value in backtrack_values:
        t = time.time()
        min_x, min_f_x, num_iter = functions.backtraking_line_search(f, f_, value, args.eps)
        delta_t = (time.time() - t) * 1000
        print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms, for {}'.format(
            min_x, min_f_x, num_iter, delta_t, value))
