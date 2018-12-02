import numpy as np

import sys
sys.path.append('../')  # Fix it later, import from other dir not working
from lista01 import functions as f1


def conjugate_gradient(f, g, H, x, eps, max_iter=15000, iter=None):
    gx = g(x if iter is None else [x, iter])
    d = -gx
    k = 0
    while (k < max_iter):
        Hx = H(x if iter is None else [x, iter])
        alpha = np.dot(gx, gx) / np.dot(np.dot(d, Hx), d)
        x += alpha * d
        fx = f(x if iter is None else [x, iter])

        if np.all(abs(alpha * d) < eps):
            return x, fx, k

        gx_new = g(x if iter is None else [x, iter])
        beta = np.dot(gx_new, gx_new) / np.dot(gx, gx)
        d = -gx_new + beta * d
        gx = gx_new
        k += 1

    return x, fx, k


def fletcher_reeves(f, g, get_f, get_f_, x, eps, max_iter=15000):
    gx = g(x)
    d = -gx
    k = 0
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

    return x, fx_star, k


def dfp(f, g, x, eps1, max_iter=15000):
    k = 0
    m = 0
    rho = 0.1
    sigma = 0.7
    tau = 0.1
    qui = 0.75
    M = 600
    eps2 = 10**(-10)

    x_size = len(x)
    S = np.eye(x_size)
    gx = g(x)
    f0 = f(x)

    m += 2
    f00 = f0
    delta_f0 = f0
    while (k < max_iter):
        # Step 2
        d = -np.dot(S, gx)
        alpha_L = 0
        alpha_U = 10**(99)
        fL = f0
        fL_ = np.dot(g(x + alpha_L * d), d)

        if abs(fL_) > eps2:
            alpha0 = -2 * delta_f0 / fL_
        else:
            alpha0 = 1

        if (alpha0 <= 0) or (alpha0 > 1):
            alpha0 = 1

        extrapolation = True
        interpolation = True
        while (extrapolation):
            while (interpolation):
                # Step 3
                delta = alpha0 * d
                f0 = f(x + delta)
                m += 1

                # Step 4
                if (f0 > fL + rho * (alpha0 - alpha_L) * fL_) and (abs(fL - f0) > eps2) and (m < M):
                    if alpha0 < alpha_U:
                        alpha_U = alpha0

                    alpha0_est = alpha_L + (
                        (alpha0 - alpha_L)**2 * fL_) / (2 * (fL - f0 + (alpha0 - alpha_L) * fL_))

                    alpha0L_est = alpha_L + tau * (alpha_U - alpha_L)
                    if alpha0_est < alpha0L_est:
                        alpha0_est = alpha0L_est

                    alpha0U_est = alpha_U - tau * (alpha_U - alpha_L)
                    if alpha0_est > alpha0U_est:
                        alpha0_est = alpha0U_est

                    alpha0 = alpha0_est
                    # Go to step 3
                else:
                    interpolation = False

            # Step 5
            f0_ = np.dot(g(x + alpha0 * d), d)
            m += 1

            # Step 6
            if (f0_ < sigma * fL_) and (abs(fL - f0) > eps2) and (m < M):
                delta_alpha0 = (alpha0 - alpha_L) * f0_ / (fL_ - f0_)

                if delta_alpha0 < 0:
                    alpha0_est = 2 * alpha0
                else:
                    alpha0_est = alpha0 + delta_alpha0

                alpha0U_est = alpha0 + qui * (alpha_U - alpha0)
                if alpha0_est > alpha0U_est:
                    alpha0_est = alpha0U_est

                alpha_L = alpha0
                alpha0 = alpha0_est
                fL = f0
                fL_ = f0_
                # Go to Step 3
                interpolation = True
            else:
                extrapolation = False

        # Step 7
        x += delta
        delta_f0 = f00 - f0
        if (((np.dot(delta, delta) < eps1) and (abs(delta_f0) < eps1)) or (m >= M)):
            return x, f(x), k
        f00 = f0

        # Step 8
        gx_new = g(x)
        gamma = gx_new - gx
        D = np.dot(delta, gamma)
        if D <= 0:
            S = np.eye(x_size)
        else:
            S = S + np.outer(delta, delta) / np.dot(delta, gamma) - np.dot(
                np.outer(np.dot(S, gamma), gamma), S) / np.dot(np.dot(gamma, S), gamma)

        gx = gx_new
        k += 1
        # Go to Step 2

    return x, f(x), k


def bfgs(f, g, x, eps1, max_iter=15000):
    k = 0
    m = 0
    rho = 0.1
    sigma = 0.7
    tau = 0.1
    qui = 0.75
    M = 600
    eps2 = 10**(-10)

    x_size = len(x)
    S = np.eye(x_size)
    gx = g(x)
    f0 = f(x)

    m += 2
    f00 = f0
    delta_f0 = f0
    while (k < max_iter):
        # Step 2
        d = -np.dot(S, gx)
        alpha_L = 0
        alpha_U = 10**(99)
        fL = f0
        fL_ = np.dot(g(x + alpha_L * d), d)

        if abs(fL_) > eps2:
            alpha0 = -2 * delta_f0 / fL_
        else:
            alpha0 = 1

        if (alpha0 <= 0) or (alpha0 > 1):
            alpha0 = 1

        extrapolation = True
        interpolation = True
        while (extrapolation):
            while (interpolation):
                # Step 3
                delta = alpha0 * d
                f0 = f(x + delta)
                m += 1

                # Step 4
                if (f0 > fL + rho * (alpha0 - alpha_L) * fL_) and (abs(fL - f0) > eps2) and (m < M):
                    if alpha0 < alpha_U:
                        alpha_U = alpha0

                    alpha0_est = alpha_L + (
                        (alpha0 - alpha_L)**2 * fL_) / (2 * (fL - f0 + (alpha0 - alpha_L) * fL_))

                    alpha0L_est = alpha_L + tau * (alpha_U - alpha_L)
                    if alpha0_est < alpha0L_est:
                        alpha0_est = alpha0L_est

                    alpha0U_est = alpha_U - tau * (alpha_U - alpha_L)
                    if alpha0_est > alpha0U_est:
                        alpha0_est = alpha0U_est

                    alpha0 = alpha0_est
                    # Go to step 3
                else:
                    interpolation = False

            # Step 5
            f0_ = np.dot(g(x + alpha0 * d), d)
            m += 1

            # Step 6
            if (f0_ < sigma * fL_) and (abs(fL - f0) > eps2) and (m < M):
                delta_alpha0 = (alpha0 - alpha_L) * f0_ / (fL_ - f0_)

                if delta_alpha0 < 0:
                    alpha0_est = 2 * alpha0
                else:
                    alpha0_est = alpha0 + delta_alpha0

                alpha0U_est = alpha0 + qui * (alpha_U - alpha0)
                if alpha0_est > alpha0U_est:
                    alpha0_est = alpha0U_est

                alpha_L = alpha0
                alpha0 = alpha0_est
                fL = f0
                fL_ = f0_
                # Go to Step 3
                interpolation = True
            else:
                extrapolation = False

        # Step 7
        x += delta
        delta_f0 = f00 - f0
        if (((np.dot(delta, delta) < eps1) and (abs(delta_f0) < eps1)) or (m >= M)):
            return x, f(x), k
        f00 = f0

        # Step 8
        gx_new = g(x)
        gamma = gx_new - gx
        D = np.dot(delta, gamma)
        if D <= 0:
            S = np.eye(x_size)
        else:
            aux = np.dot(gamma, delta)
            S = S + (1 + np.dot(np.dot(gamma, S), gamma) / aux) * (np.outer(delta, delta) / aux) - (
                np.dot(np.outer(delta, gamma), S) + np.outer(np.dot(S, gamma), delta)) / aux

        gx = gx_new
        k += 1
        # Go to Step 2

    return x, f(x), k
