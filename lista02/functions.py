import matplotlib as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.linalg


def fp(x):
    return [x[0] + 10 * x[1], 5**(0.5) * (x[2] - x[3]), (x[1] - 2 * x[2])**2, 10 * (x[0] - x[3])**2]


def J(x):
    return np.array([[1, 10, 0, 0], [0, 0, 5**(0.5), -5**(0.5)],
                     [0, 2 * x[1] - 4 * x[2], 8 * x[2] - 4 * x[1], 0],
                     [20 * x[0] - 20 * x[3], 0, 0, 20 * x[3] - 20 * x[0]]])


def plot3d_f(f, num_exe, minv, maxv):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cdict = {
        'red': ((0.0, 0.25, .25), (0.7, .59, .59), (1., 1., 1.)),
        'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
        'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
    }

    cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    x = y = np.arange(minv, maxv, 0.01)
    X, Y = np.meshgrid(x, y)
    zs = np.array([f([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=cm)  # cmap=cm.coolwarm)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')

    fig.savefig('exe{}_function.png'.format(num_exe))
    plt.show()


def steepest_descent(get_f, get_f_, g, line_search, x0, min_x, max_x, eps, max_iter=15000):
    x = x0
    k = 0
    while (k < max_iter):
        d = -0.01 * g(x)
        f = get_f(x, d)
        f_ = get_f_(x, d)

        [alpha_star, fx_star, k_star] = line_search([f, f_], [min_x, max_x], eps)
        x += alpha_star * d

        if np.all(abs(alpha_star * d) < eps):
            return x, fx_star, k
        k += 1

    print('Reached max iterations!')
    return x, fx_star, k


def steepest_descent_no_line_search(f, g, x, eps, max_iter=15000, iter=None):
    alpha = 1.0
    fx = f(x if iter is None else [x, iter])
    k = 0
    while (k < max_iter):
        gx = g(x if iter is None else [x, iter])
        d = -1 * gx

        alpha_hat = alpha
        x_ = x - alpha_hat * gx
        f_hat = f(x_ if iter is None else [x_, iter])

        alpha = (np.dot(gx, gx) * alpha**2) / (2 * (f_hat - fx + alpha * np.dot(gx, gx)))

        x += alpha * d
        fx = f(x if iter is None else [x, iter])

        if np.all(abs(alpha * d) < eps):
            return x, fx, k
        k += 1

    print('Reached max iterations!')
    return x, fx, k


def modified_newton(g, H, get_f, get_f_, line_search, x0, min_x, max_x, eps, max_iter=15000):
    x = x0
    k = 0
    while (k < max_iter):
        gx = g(x)
        Hx = H(x)

        # --- Test1 beta ---
        # if np.all(np.linalg.eigvals(Hx) > 0):
        #     # Is positive definite
        #     beta = 10**(-5)
        #     # beta = 1
        # else:
        #     beta = 10**(5)

        # Hx = (Hx + beta * np.eye(len(x))) / (1 + beta)
        # --- end Test1 beta ---

        # --- Test2 beta ---  Works amazingly better!
        if not np.all(np.linalg.eigvals(Hx) > 0):
            Hx = (Hx + np.eye(len(x))) / 2
        # --- end Test2 beta ---

        H_inv = np.linalg.inv(Hx)
        d = -1 * np.dot(H_inv, gx)

        f = get_f(x, d)
        f_ = get_f_(x, d)
        [alpha_star, fx_star, k_star] = line_search([f, f_], [min_x, max_x], eps)
        x += alpha_star * d

        min_x = alpha_star - alpha_star / 2
        max_x = alpha_star + alpha_star / 2

        if np.all(abs(alpha_star * d) < eps):
            return x, fx_star, k
        k += 1

    print('Reached max iterations!')
    return x, fx_star, k


def gauss_newton(f, get_f, get_f_, line_search, x, min_x, max_x, eps, max_iter=15000):
    k = 0
    while (k < max_iter):
        fx = f(x)
        fpx = fp(x)
        Jx = J(x)
        gx = 2 * np.dot(Jx, fpx)
        Hx = 2 * np.dot(Jx, Jx)

        Hx = matthews_davies(Hx)
        d = -np.dot(np.linalg.inv(Hx), gx)

        f_alpha = get_f(x, d)
        f_alpha_ = get_f_(x, d)
        [alpha_star, fx_star, k_star] = line_search([f_alpha, f_alpha_], [min_x, max_x], eps)

        k += 1
        x += alpha_star * d
        min_x = alpha_star - alpha_star / 2
        max_x = alpha_star + alpha_star / 2

        if np.all(abs(fx - f(x)) < eps):
            return x, f(x), k

    print('Reached max iterations!')
    return x, fx_star, k


def matthews_davies(G):
    P, L, U = scipy.linalg.lu(G)
    if np.any([U[i, i] for i in range(len(U))] < 0):
        for i in range(len(U)):
            if float(U[i, i]) == 0.0:
                U[i, i] = 1.0
            else:
                U[i, i] = abs(U[i, i])

        G = np.dot(L, U)
    return G
