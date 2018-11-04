import numpy as np


def steepest_descent(get_f, get_f_, g, line_search, x0, min_x, max_x, eps, max_iter=100000):
    x = x0
    k = 0
    while (k < max_iter):
        d = -1 * g(x)
        f = get_f(x, d)
        f_ = get_f_(x, d)

        [alpha_star, fx_star, k_star] = line_search([f, f_], [min_x, max_x], eps)
        x += alpha_star * d

        if np.all(abs(alpha_star * d) < eps):
            return x, fx_star, k
        k += 1

    print('Reached max iterations!')
    return x, fx_star, k


def steepest_descent_no_line_search(f, g, line_search, x, min_x, max_x, eps, max_iter=100):
    alpha = 1.0
    fx = f(x)
    k = 1
    while (k < max_iter):
        gx = g(x)
        d = -1 * gx

        alpha_hat = alpha
        f_hat = f(x - alpha_hat * gx)

        alpha = (np.dot(gx, gx) * alpha**2) / (2 * (f_hat - fx + alpha * np.dot(gx, gx)))

        x += alpha * d
        fx = f(x)

        if np.all(abs(alpha * d) < eps):
            return x, fx, k
        k += 1

    print('Reached max iterations!')
    return x, fx, k


def modified_newton(g, H, get_f, get_f_, line_search, x0, min_x, max_x, eps, max_iter=100):
    x = x0
    k = 0
    while (k < max_iter):
        gx = g(x)
        Hx = H(x)

        if np.all(np.linalg.eigvals(Hx) > 0):
            # Is positive definite
            beta = 10**(5)
        else:
            beta = 100**(-5)

        Hx = (Hx + beta * np.eye(len(x))) / (1 + beta)

        H_inv = np.linalg.inv(Hx)
        d = -1 * np.dot(H_inv, gx)

        f = get_f(x, d)
        f_ = get_f_(x, d)
        [alpha_star, fx_star, k_star] = line_search([f, f_], [min_x, max_x], eps)
        x += alpha_star * d

        if np.all(abs(alpha_star * d) < eps):
            return x, fx_star, k
        k += 1

    print('Reached max iterations!')
    return x, fx_star, k
