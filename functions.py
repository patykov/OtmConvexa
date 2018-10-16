import numpy as np


def dichotomos_search(f, xl, xu, uncertainty_range, max_iter=20):
    num_iter = 0
    eps = uncertainty_range / 10

    while (xu - xl > uncertainty_range):
        x = (xu + xl) / 2
        xa = x - eps / 2
        xb = x + eps / 2

        num_iter += 1
        if f(xa) < f(xb):
            xu = xb
        else:
            xl = xa

        if num_iter >= max_iter:
            break

    return x, f(x), num_iter


def fibonacci_search(f, xl, xu, uncertainty_range):
    Il = (xu - xl)
    Fn = Il / (uncertainty_range)

    F = [1, 1]
    while F[-1] < Fn:
        F.append(sum(F[-2:]) * 1.0)
    n = len(F)

    Il *= (F[n - 2] / F[n - 1])
    xa = xu - Il
    xb = xl + Il
    fa = f(xa)
    fb = f(xb)

    for k in range(1, n - 1):
        Il *= (F[n - k - 2] / F[n - k - 1])

        if fa >= fb:
            xl = xa
            xa = xb
            xb = xl + Il
            fa = fb
            fb = f(xb)
        else:
            xu = xb
            xb = xa
            xa = xu - Il
            fb = fa
            fa = f(xa)

        if xa >= xb:
            break

    return xa, fa, k


def golden_section_search(f, xl, xu, uncertainty_range, max_iter=20):
    Il = (xu - xl)
    K = 1.618034

    Il *= 1 / K
    xa = xu - Il
    xb = xl + Il
    fa = f(xa)
    fb = f(xb)

    k = 1
    while (Il >= uncertainty_range) and (xa <= xb):
        Il *= 1 / K
        k += 1

        if fa >= fb:
            xl = xa
            xa = xb
            xb = xl + Il
            fa = fb
            fb = f(xb)
        else:
            xu = xb
            xb = xa
            xa = xu - Il
            fb = fa
            fa = f(xa)

    if fa > fb:
        x = 0.5 * (xb + xu)
    elif fa == fb:
        x = 0.5 * (xa + xb)
    elif fa < fb:
        x = 0.5 * (xl + xa)

    return x, f(x), k


def quadratic_interpolation_search(f, x1, x3, uncertainty_range, max_iter=50):
    x0 = 10**99

    x2 = 0.5 * (x1 + x3)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    for k in range(1, max_iter):
        a = (x2 - x3) * f1
        b = (x3 - x1) * f2
        c = (x1 - x2) * f3

        x = (a * (x2 + x3) + b * (x3 + x1) + c * (x1 + x2)) / (2 * (a + b + c))
        fx = f(x)
        if abs(x0 - x) < uncertainty_range:
            break

        if x1 < x < x2:
            if fx <= f2:
                x3 = x2
                f3 = f2
                x2 = x
                f2 = fx
            else:
                x1 = x
                f1 = fx
        elif x2 < x < x3:
            if fx <= f2:
                x1 = x2
                f1 = f2
                x2 = x
                f2 = fx
            else:
                x3 = x
                f3 = fx
        x0 = x

    return x, fx, k


def cubic_interpolation_search(f, f_, x1, x2, x3, uncertainty_range, max_iter=50):
    x0 = 10**99

    f_1 = f_(x1)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    for k in range(1, max_iter):
        beta = (f2 - f1 + f_1 * (x1 - x2)) / ((x1 - x2)**2)
        gama = (f3 - f1 + f_1 * (x1 - x3)) / ((x1 - x3)**2)
        theta = (2 * (x1**2) - x2 * (x1 + x2)) / (x1 - x2)
        psi = (2 * (x1**2) - x3 * (x1 + x3)) / (x1 - x3)

        a3 = (beta - gama) / (theta - psi)
        a2 = beta - theta * a3
        a1 = f_1 - 2 * a2 * x1 - 3 * a3 * (x1**2)

        if (a2**2 - 3 * a1 * a3) < 0:
            raise Exception(
                'Negative square root! Initializate the algorithm with different points.')
        x_ext_1 = (-a2 - (a2**2 - 3 * a1 * a3)**(0.5)) / (3 * a3)
        x_ext_2 = (-a2 + (a2**2 - 3 * a1 * a3)**(0.5)) / (3 * a3)

        if 2 * a2 + 6 * a3 * x_ext_1 > 0:
            x = x_ext_1
        elif 2 * a2 + 6 * a3 * x_ext_2 > 0:
            x = x_ext_2

        fx = f(x)
        if abs(x0 - x) < uncertainty_range:
            break

        m = np.argmax([f1, f2, f3])
        x0 = x
        if m == 1:
            x1 = x
            f1 = fx
            f_1 = f_(x1)
        elif m == 2:
            x2 = x
            f2 = fx
        else:
            x3 = x
            f3 = fx

    return x, fx, k


def davies_swann_campey(f, x0, uncertainty_range):
    # Values suggested in the book
    delta = 0.1 * x0
    K = 0.1

    k = 0
    while (1):
        k += 1
        # Step 2
        jump = False
        k += 1
        x_1 = x0 - delta
        x1 = x0 + delta

        f0 = f(x0)
        f1 = f(x1)

        # Step 3
        if f0 > f1:
            p = 1
        else:
            f_1 = f(x_1)
            if f_1 < f0:
                p = -1
            else:
                # f_1 >= f0 <= f1
                # Go to step 7
                jump = True

        if not jump:
            # Step 4
            last_f = f0
            last_2f = f0
            last_x = x0
            exp2 = 1
            while (1):
                new_f = f(last_x + (exp2 * p * delta))
                if new_f > last_f:
                    break

                last_2f = last_f
                last_f = new_f
                last_x += exp2 * p * delta
                exp2 *= 2

            # Step 5
            xm = last_x + (exp2 / 2 * p * delta)
            fm = f(xm)

            # Step 6
            if fm >= last_f:
                if last_2f - fm == 0:
                    x0 = x_1
                else:
                    x0 = x_1 + (exp2 / 4) * (p * delta *
                                             (last_2f - fm)) / (last_2f - 2 * last_f + fm)

            else:
                if last_f - new_f == 0:
                    x0 = xm
                else:
                    x0 = xm + (exp2 / 4) * (p * delta *
                                            (last_f - new_f)) / (last_f - 2 * fm + new_f)

            if exp2 / 2 * delta < uncertainty_range:
                return x0, f(x0), k
            else:
                delta *= K
                # Go to step 2
                continue

        # Step 7
        if f_1 - f1 != 0:
            x0 += (delta * (f_1 - f1)) / (2 * (f_1 - 2 * f0 + f1))

        if delta < uncertainty_range:
            return x0, f(x0), k
        else:
            delta *= K


def backtraking_line_search(f, f_, x, uncertainty_range):
    alfa = 0.2
    beta = 0.1

    t = 1
    k = 1
    while (1):
        delta_x = -1 * np.sign(f_(x))
        while f(x + t * delta_x) > (f(x) + alfa * t * np.transpose(f_(x)) * delta_x):
            t *= beta
            k += 1

        x_min = x + t * delta_x
        if abs(x - x_min) < uncertainty_range:
            return x_min, f(x_min), k
        else:
            x = x_min
