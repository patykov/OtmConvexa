import numpy as np

import sys
sys.path.append('../')  # Fix it later, import from other dir not working
from lista01 import functions as f1


def gradient_descent(f, g, H, x, eps, max_iter=15000):
    gx = g(x)
    d = -gx
    k = 1
    while (k < max_iter):
        Hx = H(x)
        alpha = np.dot(gx, gx) / np.dot(np.dot(d, Hx), d)
        x += alpha * d
        fx = f(x)

        if np.all(abs(alpha * d) < eps):
            return x, fx, k

        gx_new = g(x)
        beta = np.dot(gx_new, gx_new) / np.dot(gx, gx)
        d = -gx_new + beta * d
        gx = gx_new
        k += 1


def fletcher_reeves(f, g, get_f, get_f_, x, eps, max_iter=15000):
    gx = g(x)
    d = -gx
    k = 1
    while (k < max_iter):
        f = get_f(x, d)
        f_ = get_f_(x, d)
        [alpha_star, fx_star, k_star] = f1.backtraking_line_search([f, f_], x, eps)

        x += alpha_star * d
        if np.all(abs(alpha_star * d) < eps):
            return x, fx_star, k

        gx_new = g(x)
        beta = np.dot(gx_new, gx_new) / np.dot(gx, gx)
        d = -gx_new + beta * d
        gx = gx_new
        k += 1
