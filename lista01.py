import argparse

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

    functions.dichotomos_search(f, -0.5, 0.5, 10**(-5))
