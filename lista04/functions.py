import sys

sys.path.append('../')  # Fix it later, import from other dir not working
from lista01 import functions as f1
from lista02 import functions as f2
from lista03 import functions as f3


def iterative_steepest_descent(set_f_t, set_g_t, orig, z, eps, t=0.001, mu=10):
    m = 4

    all_k = 0
    count = 0
    while t < 10**8:
        # Getting f and g for new t
        f = set_f_t(t)
        g = set_g_t(t)
        z_opt, _, k = f2.steepest_descent_no_line_search(f, g, z, eps)

        all_k += k
        count += 1
        x, fx = orig(z)
        if (m / t) < eps:
            return x, fx, all_k, count

        t = mu * t
        z = z_opt

    return x, fx, all_k, count


def iterative_conjugate_gradient(set_f_t, set_g_t, set_H_t, orig, z, eps, t=1, mu=1.2):
    m = 4

    all_k = 0
    count = 0
    while t < 10**8:
        # Getting f and g for new t
        f = set_f_t(t)
        g = set_g_t(t)
        H = set_H_t(t)
        z, _, k = f3.conjugate_gradient(f, g, H, z, eps)

        all_k += k
        count += 1
        x, fx = orig(z)
        if (m / t) < eps:
            return x, fx, all_k, count

        t *= mu

    return x, fx, all_k, count


def iterative_quasi_newton(set_f_t, set_g_t, orig, z, eps, t=1, mu=1.2):
    m = 4

    all_k = 0
    count = 0
    while t < 10**8:
        # Getting f and g for new t
        f = set_f_t(t)
        g = set_g_t(t)
        z, _, k = f3.dfp(f, g, z, eps)

        all_k += k
        count += 1
        x, fx = orig(z)
        if (m / t) < eps:
            return x, fx, all_k, count

        t *= mu

    return x, fx, all_k, count


def iterative_newton(set_g_t, set_H_t, set_get_f_t, set_get_g_t, orig, z, eps, t=1, mu=1.2):
    m = 4

    all_k = 0
    count = 0
    while t < 10**8:
        g = set_g_t(t)
        H = set_H_t(t)
        get_f = set_get_f_t(t)
        get_g = set_get_g_t(t)
        z, _, k = f2.modified_newton(g, H, get_f, get_g, f1.backtraking_line_search, z, 0, 1, eps)
        all_k += k
        count += 1

        x, fx = orig(z)
        if (m / t) < eps:
            return x, fx, all_k, count

        t *= mu

    return x, fx, all_k, count
