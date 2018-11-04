import argparse
import time
import functions

import sys
sys.path.append('../')  # Fix it later, import from other dir not working
from lista01 import functions as line_search


class MyFunction(object):
    def __init__(self, exe_num):
        self.exe_num = exe_num

    def get_f(self, x0, d):
        if self.exe_num == 5.7:

            def f(alpha):
                x = x0 + alpha * d
                return (x[0]**2 + x[1]**2 - 1)**2 + (x[0] + x[1] - 1)**2

        return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    parser.add_argument('-min', help='Initial min value.', required=True, type=float)
    parser.add_argument('-max', help='Initial max value.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    points = [[4, 4], [4, -4], [-4, 4], [-4, -4]]

    if args.exe_num == 5.7:
        for x in points:
            print('\n\nInitial point: ' + str(x))
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                line_search.quadratic_interpolation_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent(
                line_search.backtraking_line_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

    elif args.exe_num == 5.8:
        for x in points:
            print('\n\nInitial point: ' + str(x))
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent_no_line_search(
                line_search.golden_section_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent_no_line_search(
                line_search.quadratic_interpolation_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.steepest_descent_no_line_search(
                line_search.backtraking_line_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

    elif args.exe_num == 5.17:
        for x in points:
            print('\n\nInitial point: ' + str(x))
            print('Golden Section Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(line_search.golden_section_search,
                                                                  x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Quadratic Interpolation Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                line_search.quadratic_interpolation_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))

            print('Backtraking Line Search')
            t = time.time()
            [x_min, fx_min, num_iter] = functions.modified_newton(
                line_search.backtraking_line_search, x, args.min, args.max, eps)
            delta_t = (time.time() - t) * 1000
            print('x: [{:.5f}, {:.5f}], f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
                x_min[0], x_min[1], fx_min, num_iter, delta_t))
