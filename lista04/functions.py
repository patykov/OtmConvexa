import numpy as np

import sys
sys.path.append('../')  # Fix it later, import from other dir not working
from lista03 import functions as f3
from lista02 import functions as f2


def iterative_steepest_descent(f, g, orig, z, eps):
    m = 4
    t = 1
    mu = 1.2
    z, _, k = f2.steepest_descent_no_line_search(f, g, z, eps, iter=t)
    x, fx = orig(z)
    all_k = 0
    count = 0
    while t < 10**8:
        z, _, k = f2.steepest_descent_no_line_search(f, g, z, eps, iter=t)
        all_k += k
        count += 1
        x, fx = orig(z)
        if abs(m / t) < eps:
            return x, fx, all_k, count

        t *= mu

    return x, fx, all_k, count


def iterative_conjugate_gradient(f, g, H, orig, z, eps):
    m = 4
    t = 1
    mu = 1.2
    z, _, k = f3.conjugate_gradient(f, g, H, z, eps, iter=t)
    x, fx = orig(z)
    all_k = 0
    count = 0
    while t < 10**8:
        z, _, k = f3.conjugate_gradient(f, g, H, z, eps, iter=t)
        all_k += k
        count += 1
        x, fx = orig(z)
        if abs(m / t) < eps:
            return x, fx, all_k, count

        t *= mu

    return x, fx, all_k, count
