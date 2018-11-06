import argparse
import time
import functions
import numpy as np

import sys
sys.path.append('../')  # Fix it later, import from other dir not working
from lista01 import functions as line_search


def get_defined_functions(exe_num):
    if exe_num in [5.7, 5.8, 5.17]:

        def f(x):
            return (x[0]**2 + x[1]**2 - 1)**2 + (x[0] + x[1] - 1)**2

        def g(x):
            a = (x[0]**2 + x[1]**2 - 1)
            b = 2 * x[0] + 2 * x[1] - 2
            return np.array([4 * x[0] * a + b, 4 * x[1] * a + b])

        def H(x):
            a = (x[0]**2 + x[1]**2 - 1)
            b = 6 * x[0] * x[1] + 2
            return np.array([[4 * a + 6 * x[0] + 2, b], [b, 4 * a + 6 * x[1] + 2]])

        def get_f(x0, d):
            def f(alpha):
                x = x0 + alpha * d
                return (x[0]**2 + x[1]**2 - 1)**2 + (x[0] + x[1] - 1)**2

            return f

        def get_f_(x0, d):
            def f_(alpha):
                x = x0 + alpha * d
                a = (x[0]**2 + x[1]**2 - 1)
                da_dalpha = x[0] * d[0] + alpha * (d[0]**2 + d[1]**2) + x[1] * d[1]
                b = x[0] + x[1] - 1
                db_dalpha = d[0] + d[1]
                return 4 * a * da_dalpha + 2 * b * db_dalpha

            return f_

    elif exe_num == 5.20:

        def f(x):
            return (x[0] + 10 * x[1])**2 + 5 * (x[2] - x[3])**2 + (
                x[1] - 2 * x[2])**4 + 100 * (x[0] - x[3])**4

        def get_f(x0, d):
            def f(alpha):
                x = x0 + alpha * d
                return (x[0] + 10 * x[1])**2 + 5 * (x[2] - x[3])**2 + (
                    x[1] - 2 * x[2])**4 + 100 * (x[0] - x[3])**4

            return f

        def get_f_(x0, d):
            def f_(alpha):
                x = x0 + alpha * d
                return 2 * (x[0] + 10 * x[1]) * (d[0] + 10 * d[1]) + 10 * (
                    x[2] - x[3]) * (d[2] - d[3]) + 4 * (x[1] - 2 * x[2])**3 * (
                    d[1] - 2 * d[2]) + 400 * (x[0] - x[3])**3 * (d[0] - d[3])

            return f_

        def g(x):
            a = 400 * (x[0] - x[3])**3
            b = (x[1] - 2 * x[2])**3
            return np.array([
                2 * x[0] + 20 * x[1] + a, 20 * x[0] + 200 * x[1] + 4 * b,
                10 * x[2] - 10 * x[3] - 8 * b, 10 * x[3] - 10 * x[2] - a
            ])

        def H(x):
            a = 1200 * (x[0] - x[3])**2
            b = (x[1] - 2 * x[2])**2
            return np.array([
                [a + 2, 20, 0, -1 * a], [20, 12 * b + 200, -24 * b, 0],
                [0, -24 * b, 48 * b + 10, -10], [-1 * a, 0, -10, a + 10]
            ])

    return f, g, H, get_f, get_f_


def get_points(exe_num):
    if exe_num in [5.7, 5.8, 5.17]:
        return [[4.0, 4.0], [4.0, -4.0], [-4.0, 4.0], [-4.0, -4.0]]
    elif exe_num == 5.20:
        return [[-2.0, -1.0, 1.0, 2.0], [200.0, -200.0, 100.0, -100.0]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    parser.add_argument('-min', help='Initial min value.', required=True, type=float)
    parser.add_argument('-max', help='Initial max value.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    points = get_points(args.exe_num)

    f, g, H, get_f, get_f_ = get_defined_functions(args.exe_num)

    if args.exe_num == 5.7:
        print('Exercise 5.8 - Steepest Descent')
        for x in points:
            print('\nInitial point: ' + str(x))
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                get_f, get_f_, g, line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                get_f, get_f_, g, line_search.quadratic_interpolation_search, x, args.min, args.max,
                eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                get_f, get_f_, g, line_search.backtraking_line_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

    elif args.exe_num == 5.8:
        print('Exercise 5.8 - Steepest Descent without line search')
        for x in points:
            print('\nInitial point: ' + str(x))
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent_no_line_search(
                f, g, line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent_no_line_search(
                f, g, line_search.quadratic_interpolation_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent_no_line_search(
                f, g, line_search.backtraking_line_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

    elif args.exe_num == 5.17:
        print('Exercise 5.17 - Modified Newton')
        for x in points:
            print('\nInitial point: ' + str(x))
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                g, H, get_f, get_f_, line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                g, H, get_f, get_f_, line_search.quadratic_interpolation_search, x, args.min,
                args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                g, H, get_f, get_f_, line_search.backtraking_line_search, x, args.min, args.max,
                eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

    elif args.exe_num == 5.20:
        print('Exercise 5.20')
        for x in points:
            print('\nInitial point: ' + str(x))
            print('---- Steepest Descent -----')
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                get_f, get_f_, g, line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                get_f, get_f_, g, line_search.quadratic_interpolation_search, x, args.min, args.max,
                eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                get_f, get_f_, g, line_search.backtraking_line_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f}ms \
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('\n---- Modified Newton -----')
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                g, H, get_f, get_f_, line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                g, H, get_f, get_f_, line_search.quadratic_interpolation_search, x, args.min,
                args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                g, H, get_f, get_f_, line_search.backtraking_line_search, x, args.min, args.max,
                eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('\n---- Gauss Newton -----')
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.gauss_newton(
                f, get_f, get_f_, line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.gauss_newton(
                f, get_f, get_f_, line_search.quadratic_interpolation_search, x, args.min,
                args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.gauss_newton(
                f, get_f, get_f_, line_search.backtraking_line_search, x, args.min, args.max,
                eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}, {:.5f}, {:.5f}], f(x): {:.5e}, num_iter: {}, time: {:.5f} ms\
            \n'.format(x_min[0], x_min[1], x_min[2], x_min[3], fx_min, num_iter, delta_t))
