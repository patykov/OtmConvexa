import argparse
import time

import functions


def get_f(num):
    if num == 4.2:
        def f(x):
            return -5*x**5 + 4*x**4 - 12*x**3 + 11*x**2 - 2*x + 1

    return f


def get_f_(num):
    if num == 4.2:
        def f_(x):
            return -25*x**4 + 16*x**3 - 36*x**2 + 22*x - 2
    return f_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-exe_num',
        help='Number of the exercise.',
        required=True, type=float)
    args = parser.parse_args()

    # Exercicio 4.2
    f = get_f(args.exe_num)
    f_ = get_f_(args.exe_num)

    t1 = time.time()
    min_x, min_f_x, num_iter = functions.dichotomos_search(f, -0.5, 0.5, 10**(-5))
    delta_t1 = (time.time() - t1)*1000
    print('Exercise 4.2: Dichotomous Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t1))

    t2 = time.time()
    min_x, min_f_x, num_iter = functions.fibonacci_search(f, -0.5, 0.5, 10**(-5))
    delta_t2 = (time.time() - t2)*1000
    print('Exercise 4.2: Fibonacci Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t2))

    t3 = time.time()
    min_x, min_f_x, num_iter = functions.golden_section_search(f, -0.5, 0.5, 10**(-5))
    delta_t3 = (time.time() - t3)*1000
    print('Exercise 4.2: Golden Section Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t3))

    t4 = time.time()
    min_x, min_f_x, num_iter = functions.quadratic_interpolation_search(f, -0.5, 0.5, 10**(-5))
    delta_t4 = (time.time() - t4)*1000
    print('Exercise 4.2: Quadratic Interpolation Search')
    print('x: {:.5f}, f(x): {:.5f}, num_iter: {}, time: {:.5f} ms\n'.format(
        min_x, min_f_x, num_iter, delta_t4))
