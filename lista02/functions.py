import numpy as np


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


def steepest_descent(line_search, x0, min_x, max_x, eps, max_iter=100):
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


def steepest_descent_no_line_search(line_search, x, min_x, max_x, eps, max_iter=100):
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


def modified_newton(line_search, x0, min_x, max_x, eps, max_iter=100):
    x = x0
    beta = 1
    k = 0
    while (k < max_iter):
        gx = g(x)
        Hx = H(x)

        if not np.all(np.linalg.eigvals(Hx) > 0):
            Hx = (H(x) + beta * np.eye(len(x))) / (1 + beta)

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
