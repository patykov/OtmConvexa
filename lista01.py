import argparse
import time

import functions


def get_f(num):
    if num == 4.2:

        def f(x):
            return -5 * x**5 + 4 * x**4 - 12 * x**3 + 11 * x**2 - 2 * x + 1

    return f


def get_f_(num):
    if num == 4.2:

        def f_(x):
            return -25 * x**4 + 16 * x**3 - 36 * x**2 + 22 * x - 2

    return f_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    args = parser.parse_args()

    # Exercicio 4.2
    f = get_f(args.exe_num)
    f_ = get_f_(args.exe_num)

    print('Exercise {}:\n'.format(args.exe_num))
    t = time.time()
    min_x, min_f_x, num_iter = functions.dichotomos_search(f, -0.5, 0.5, 10**(-5))
    delta_t = (time.time() - t) * 1000
    print('Dichotomous Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    min_x, min_f_x, num_iter = functions.fibonacci_search(f, -0.5, 0.5, 10**(-5))
    delta_t = (time.time() - t) * 1000
    print('Fibonacci Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    min_x, min_f_x, num_iter = functions.golden_section_search(f, -0.5, 0.5, 10**(-5))
    delta_t = (time.time() - t) * 1000
    print('Golden Section Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    min_x, min_f_x, num_iter = functions.quadratic_interpolation_search(f, -0.5, 0.5, 10**(-5))
    delta_t = (time.time() - t) * 1000
    print('Quadratic Interpolation Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    # RUN WITH DIFFERENT POINTS X1, X2, X3
    min_x, min_f_x, num_iter = functions.cubic_interpolation_search(f, f_, -0.2, 0.1, 0.2, 10**(-5))
    delta_t = (time.time() - t) * 1000
    print('Cubic Interpolation Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    # RUN WITH DIFFERENT POINTS X0
    min_x, min_f_x, num_iter = functions.davies_swann_campey(f, 0.2, 10**(-5))
    delta_t = (time.time() - t) * 1000
    print('Davies, Swann and Campey Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t))

    t = time.time()
    min_x, min_f_x, num_iter = functions.backtraking_line_search(f, f_, -0.5, 10**(-5))
    delta_t = (time.time() - t) * 1000
    print('Backtraking Line Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t))
