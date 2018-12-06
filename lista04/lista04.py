import argparse
import time

import numpy as np
from scipy.linalg import null_space

import functions


def restricted_log(x):
    if x <= 0.0:
        return -np.inf
    else:
        return np.log(x)


def get_defined_functions(exe_num):
    if exe_num == 1:
        A = np.array([[1, 2, 1, 2], [1, 1, 2, 4]])
        b = np.array([3, 5])
        c = np.array([1, 1.5, 1, 1])

        F = null_space(A)

        # Getting initial x
        x0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        def set_f_t(t):
            def f(z):
                f0 = t * np.dot(c, (np.dot(F, z) + x0))
                I_ = [restricted_log(np.dot(F[i, :], z) + x0[i]) for i in range(4)]

                return f0 - sum(I_)

            return f

        def set_get_f_t(t):
            def get_f(z0, d):
                def f(alpha):
                    z = z0 + alpha * d
                    f0 = np.dot((t * c), (np.dot(F, z) + x0))
                    I_ = [restricted_log(np.dot(F[i, :], z) + x0[i]) for i in range(4)]

                    return f0 - sum(I_)

                return f

            return get_f

        def set_g_t(t):
            def g(z):
                logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
                logs_dev = np.array([
                    sum([F[i, 0] / logs_divs[i] for i in range(4)]),
                    sum([F[i, 1] / logs_divs[i] for i in range(4)])
                ])
                f0_dev = np.array([
                    sum([F[i, 0] * t * np.conj(c[i]) for i in range(4)]),
                    sum([F[i, 1] * t * np.conj(c[i]) for i in range(4)])
                ])

                return f0_dev - logs_dev

            return g

        def set_get_g_t(t):
            def get_g(z0, d):
                def g(alpha):
                    z = z0 + alpha * d
                    logs_dev = sum([
                        np.dot(F[i, :], z) / (np.dot(F[i, :], z) + x0[i]) for i in range(4)])
                    f0_dev = t * np.dot(c, np.dot(F, d))
                    return f0_dev - logs_dev

                return g

            return get_g

        def set_H_t(t):
            def H(z):
                logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
                logs_dev = np.array([[F[i, 0] / logs_divs[i] for i in range(4)],
                                    [F[i, 1] / logs_divs[i] for i in range(4)]])
                aux = logs_dev**2
                dz1z1 = sum(aux[0])
                dz2z2 = sum(aux[1])
                dz1z2 = sum([(F[i, 0] * F[i, 1]) / logs_divs[i]**2 for i in range(4)])

                return np.array([[dz1z1, dz1z2], [dz1z2, dz2z2]])
            return H

        def original(z):
            min_x = np.dot(F, z) + x0
            min_fx = np.dot(c, min_x)

            return min_x, min_fx

        return x0, set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original

    elif exe_num == 2:
        def c1(x):
            A = [[0.25, 0], [0, 1.0]]
            b = [0.5, 0]
            aux1 = -1.0 * np.dot(np.dot(x, A), x)
            aux2 = np.dot(x[:2], b)

            return aux1 + aux2 + 3.0/4.0

        def c2(x):
            C = [[5.0, 3.0], [3.0, 5.0]]
            d = [11/2.0, 13/2.0]
            aux1 = -1.0/8 * np.dot(np.dot(x, C), x)
            aux2 = np.dot(x, d)

            return aux1 + aux2 - 35/2.0

        def set_f_t(t):
            def f(x):
                aux1 = (x[0] - x[2])**2 + (x[1] - x[3])**2
                aux2 = restricted_log(c1(x[:2])) + restricted_log(c2(x[2:]))

                return aux1 - aux2

            return f

        def set_get_f_t(t):
            def get_f(x0, d):
                def f(alpha):
                    x = x0 + alpha * d
                    aux1 = (x[0] - x[2])**2 + (x[1] - x[3])**2
                    aux2 = restricted_log(c1(x[:2])) + restricted_log(c2(x[2:]))

                    return aux1 - aux2

                return f

            return get_f

        def set_g_t(t):
            def g(x):
                logs_div = [1/c1(x[:2]), 1/c2(x[2:])]
                dx0 = 2*t*(x[0] - x[2]) - logs_div[0] * (-0.5*x[0] + 0.5)
                dx1 = 2*t*(x[1] - x[3]) - logs_div[0] * (-2*x[1])
                dx2 = -2*t*(x[0] - x[2]) - logs_div[1] * (-(5/4.0)*x[2] - (3/4.0)*x[3] + 11/2.0)
                dx3 = -2*t*(x[1] - x[3]) - logs_div[1] * (-(3/4.0)*x[2] - (5/4.0)*x[3] + 13/2.0)

                return np.array([dx0, dx1, dx2, dx3])

            return g

        def set_get_g_t(t):
            def get_g(x0, d):
                def g(alpha):
                    x = x0 + alpha * d

                    ddist = 2*t*((x[0] - x[2])*(d[0] - d[2]) + (x[1] - x[3])*(d[1] - d[3]))
                    dc1 = (0.5*x[0]*d[0] + 2*x[1]*d[1] - 0.5*d[0])/c1(x[:2])
                    dc2 = ((5/4.0)*x[2]*d[2] + (3/4.0)*(
                        x[2]*d[3] + d[2]*x[3]) + (5/4.0)*x[3]*d[3] - (11/2.0)*d[2] - (13/2.0)*d[3]
                        )/c2(x[2:])

                    return ddist + dc1 + dc2

                return g

            return get_g

        def set_H_t(t):
            def H(x):
                logs_div = [1/c1(x[:2]), 1/c2(x[2:])]
                dc1x1 = -0.5*x[0] + 0.5
                dc1x2 = -2*x[1]
                dc2x3 = -(5/4.0)*x[2] - (3/4.0)*x[3] + 11/2.0
                dc2x4 = -(3/4.0)*x[2] - (5/4.0)*x[3] + 13/2.0

                dx1x1 = 2*t + (logs_div[0]*dc1x1)**2 + 0.5*logs_div[0]
                dx1x2 = logs_div[0]**2 * dc1x1*dc1x2
                dx1x3 = -2*t
                dx1x4 = 0

                dx2x2 = 2*t + (logs_div[0]*dc1x2)**2 + 2*logs_div[0]
                dx2x3 = 0
                dx2x4 = -2*t

                dx3x3 = 2*t + (logs_div[1]*dc2x3)**2 + (5/4.0)*logs_div[1]
                dx3x4 = logs_div[1]**2 * dc2x3*dc2x4 + (3/4.0)*logs_div[1]

                dx4x4 = 2*t + (logs_div[1]*dc2x4)**2 + (5/4.0)*logs_div[1]

                return np.array([
                    [dx1x1, dx1x2, dx1x3, dx1x4], [dx1x2, dx2x2, dx2x3, dx2x4],
                    [dx1x3, dx2x3, dx3x3, dx3x4], [dx1x4, dx2x4, dx3x4, dx4x4]
                    ])
            return H

        def original(x):
            x = np.array(x)
            return x, np.linalg.norm(x[:2] - x[2:])

        return set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original

    elif exe_num == 3:
        A = np.array([[1, 1, 1]])
        b = np.array([3])
        C = np.array([[4, 0, 0], [0, 1, -1], [0, -1, 1]])
        d = np.array([-8, -6, -6])

        F = null_space(A)

        # Getting initial x
        x0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        def set_f_t(t):
            def f(z):
                f0 = np.dot(np.dot(
                    0.5*(np.dot(F, z) + x0), C), np.dot(F, z) + x0) + np.dot(np.dot(F, z) + x0, d)
                I_ = [restricted_log(np.dot(F[i, :], z) + x0[i]) for i in range(3)]

                return t*f0 - sum(I_)

            return f

        def set_get_f_t(t):
            def get_f(z0, d_):
                def f(alpha):
                    z = z0 + alpha * d_
                    f0 = np.dot(np.dot(0.5*(np.dot(F, z) + x0), C), np.dot(F, z) + x0) + np.dot(
                        np.dot(F, z) + x0, d)
                    I_ = [restricted_log(np.dot(F[i, :], z) + x0[i]) for i in range(3)]

                    return f0 - sum(I_)

                return f

            return get_f

        def set_g_t(t):
            def g(z):
                z_conj = np.conj(z)
                F_conj = np.conj(F)
                x0_conj = np.conj(x0)

                df01 = [
                    sum([F[i, j]*(
                        C[i, 0]*(x0_conj[i] + F_conj[i, 0]*z_conj[0] + F_conj[i, 1]*z_conj[1]))/2.0
                        for i in range(3)]) for j in range(2)]
                df02 = [sum([d[i]*F_conj[i, j] for i in range(3)])
                        for j in range(2)]
                df03 = [x0[i] + F[i, 0]*z[0] + F[i, 1]*z[1] for i in range(3)]
                df04 = [sum([
                    df03[i]*(C[0, i]*F_conj[0, j] + C[1, i]*F_conj[1, j] + C[2, i]*F_conj[2, j]
                    )/2.0 - F[i, j]/df03[i] for i in range(3)])
                        for j in range(2)]

                return np.array(np.array(df01) + np.array(df02) + np.array(df04))

            return g

        def set_get_g_t(t):
            def get_g(z0, d_):
                def g(alpha):
                    z = z0 + alpha * d_
                    z_conj = np.conj(z)
                    F_conj = np.conj(F)
                    x0_conj = np.conj(x0)

                    df01 = [
                        sum([F[i, j]*(
                            C[i, 0]*(
                                x0_conj[i] + F_conj[i, 0]*z_conj[0] + F_conj[i, 1]*z_conj[1]))/2.0
                            for i in range(3)]) for j in range(2)]

                    df02 = [sum([d[i]*F_conj[i, j] for i in range(3)])
                            for j in range(2)]
                    df03 = [x0[i] + F[i, 0]*z[0] + F[i, 1]*z[1] for i in range(3)]
                    df04 = [sum([
                        df03[i]*(C[0, i]*F_conj[0, j] + C[1, i]*F_conj[1, j] + C[2, i]*F_conj[2, j]
                        )/2.0 - F[i, j]/df03[i] for i in range(3)])
                            for j in range(2)]

                    return np.array(
                        np.array(df01) + np.array(df02) + np.array(df04))
                return g

            return get_g

        def set_H_t(t):
            def H(z):
                F_conj = np.conj(F)
                der1 = np.array([
                    [sum([(C[i, j]*(F_conj[i, 0]))/2.0 for i in range(3)]),
                     sum([(C[i, j]*(F_conj[i, 1]))/2.0 for i in range(3)])] for j in range(3)])
                der2 = [(x0[i] + F[i, 0]*z[0] + F[i, 1]*z[1])**2 for i in range(3)]

                dz1z1 = 2*sum(
                    [F[i, 0]*der1[i, 0] for i in range(3)]) + sum(
                        [F[i, 0]**2/der2[i] for i in range(3)])
                dz1z2 = sum(F[i, 1]*der1[i, 0] + F[i, 0]*der1[i, 1] for i in range(3)) + sum([
                            F[i, 0]*F[i, 1]/der2[i] for i in range(3)])
                dz2z2 = 2*sum(
                    [F[i, 1]*der1[i, 1] for i in range(3)]) + sum(
                        [F[i, 1]**2/der2[i] for i in range(3)])

                return t*np.array([[dz1z1, dz1z2], [dz1z2, dz2z2]])

            return H

        def original(z):
            min_x = np.dot(F, z) + x0
            min_fx = 0.5*np.dot(np.dot(min_x, C), min_x) + np.dot(min_x, d)

            return min_x, min_fx

        return x0, set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original

    elif exe_num == 4:
        F0 = np.array([
            [0.5, 0.55, 0.33, 2.38],
            [0.55, 0.18, -1.18, -0.4],
            [0.33, -1.18, -0.94, 1.46],
            [2.38, -0.4, 1.46, 0.17]])

        F1 = np.array([
            [5.19, 1.54, 1.56, -2.8],
            [1.54, 2.2, 0.39, -2.5],
            [1.56, 0.39, 4.43, 1.77],
            [-2.8, -2.5, 1.77, 4.06]])

        F2 = np.array([
            [-1.11, 0, -2.12, 0.38],
            [0, 1.91, -0.25, -0.58],
            [-2.12, -0.25, -1.49, 1.45],
            [0.38, -0.58, 1.45, 0.63]])

        F3 = np.array([
            [2.69, -2.24, -0.21, -0.74],
            [-2.24, 1.77, 1.16, -2.01],
            [-0.21, 1.16, -1.82, -2.79],
            [-0.74, -2.01, -2.79, -2.22]])

        F4 = np.array([
            [0.58, -2.19, 1.69, 1.28],
            [-2.19, -0.05, -0.01, 0.91],
            [1.69, -0.01, 2.56, 2.14],
            [1.28, 0.91, 2.14, -0.75]])

        c = np.array([1, 0, 2, -1])

        F = [F1, F2, F3, F4]

        x0 = np.zeros(4)
        while np.linalg.eigvalsh(F0 + sum([x0[i]*F[i] for i in range(4)]))[0] <= 0:
            x0 = np.random.randn(4)

        def set_f_t(t):
            def f(x):
                f0 = t*np.dot(c, x)
                I_ = restricted_log(np.linalg.eigvalsh(F0 + sum([x[i]*F[i] for i in range(4)]))[0])

                return f0 - I_

            return f

        def set_get_f_t(t):
            def get_f(x0, d):
                def f(alpha):
                    x = x0 + alpha * d
                    f0 = t*np.dot(c, x)
                    I_ = restricted_log(
                        np.linalg.eigvalsh(F0 + sum([x[i]*F[i] for i in range(4)]))[0])

                    return f0 - I_

                return f

            return get_f

        def set_g_t(t):
            def g(x):
                F_sum = np.matrix(F0 + sum([x[i]*F[i] for i in range(4)]))
                F_adf = np.linalg.inv(F_sum) * np.linalg.det(F_sum)
                F_dev = [np.trace(np.dot(F_adf, F[i])) for i in range(4)]
                log_dev = 1.0/np.linalg.det(F_sum)

                return np.array([t*c[i] - log_dev*F_dev[i] for i in range(4)])

            return g

        def set_get_g_t(t):
            def get_g(x0, d):
                def g(alpha):
                    x = x0 + alpha * d
                    F_sum = F0 + sum([x[i]*F[i] for i in range(4)])
                    F_adf = np.linalg.inv(F_sum) * np.linalg.det(F_sum)
                    F_dev = [np.trace(F_adf*(d[i]*F[i])) for i in range(4)]
                    log_dev = 1.0/np.linalg.det(F_sum)

                    return t*np.dot(c, d) - sum([log_dev*F_dev[i] for i in range(4)])

                return g

            return get_g

        def set_H_t(t):
            def H(x):
                F_sum = F0 + sum([x[i]*F[i] for i in range(4)])
                F_adj = np.linalg.inv(F_sum) * np.linalg.det(F_sum)
                F_det = np.linalg.det(F_sum)
                F_dev = [np.trace(np.dot(F_adj, F[i])) for i in range(4)]
                der1 = np.dot(F_dev, F_dev)/F_det

                X_list = [F_adj.dot(Fi) for Fi in F]
                der2 = np.array([[-np.trace(Xi)*np.trace(Xj) - np.sum(Xi.T * Xj)
                                  for Xi in X_list] for Xj in X_list])
                return (- der1 - der2)/F_det
            return H

        def original(x):
            return x, np.dot(c, x)

        return x0, set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original

    elif exe_num == 5:

        def set_f_t(t):
            def f(x):
                f0 = 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + 90*(x[2]**2 - x[3])**2 + (
                    x[2] - 1)**2 + 10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8*(x[1] - 1)*(x[3] - 1)
                I_ = sum(
                    [restricted_log(x[i] + 10) for i in range(4)]) + sum(
                    [restricted_log(-x[i] + 10) for i in range(4)])

                return t*f0 - I_

            return f

        def set_get_f_t(t):
            def get_f(x0, d):
                def f(alpha):
                    x = x0 + alpha * d
                    f0 = 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + 90*(x[2]**2 - x[3])**2 + (
                        x[2] - 1)**2 + 10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8*(
                        x[1] - 1)*(x[3] - 1)
                    I_ = sum(
                        [restricted_log(x[i] + 10) for i in range(4)]) + sum(
                        [restricted_log(-x[i] + 10) for i in range(4)])

                    return t*f0 - I_

                return f

            return get_f

        def set_g_t(t):
            def g(x):
                return np.array([
                    -1.0/(x[0] - 10) - 1/(x[0] + 10) - t*(400*x[0]*(- x[0]**2 + x[1]) - 2*x[0] + 2),
                    t*(- 200*x[0]**2 + (1101*x[1])/5 + (99*x[3])/5 - 40) - 1/(x[1] - 10) - 1/(
                        x[1] + 10),
                    -1.0/(x[2] - 10) - 1/(x[2] + 10) - t*(360*x[2]*(- x[2]**2 + x[3]) - 2*x[2] + 2),
                    t*(- 180*x[2]**2 + (99*x[1])/5 + (1001*x[3])/5 - 40) - 1/(x[3] - 10) - 1/(
                        x[3] + 10)
                    ])

            return g

        def set_get_g_t(t):
            def get_g(x0, d):
                def g(alpha):
                    x = x0 + alpha * d
                    d_f0 = 200*(x[0]**2 - x[1])*(2*x[0]*d[0] - d[1]) + 2*(x[0] - 1)*d[0] + 180*(
                        x[2]**2 - x[3])*(2*x[2]*d[2] - d[3]) + 2*(x[2] - 1)*d[2] + 10.1*(2*(
                            x[1] - 1)*d[1] + 2*(x[3] - 1)*d[3]) + 19.8*(d[1]*(x[3] - 1) + (
                                x[1] - 1)*d[3])
                    d_logs = sum([d[i]*(1.0/(-x[i] - 10) - 1.0/(x[i] - 10)) for i in range(4)])

                    return d_f0 + d_logs

                return g

            return get_g

        def set_H_t(t):
            def H(x):
                dx1x1 = 1/(x[0] - 10)**2 + 1/(x[0] + 10)**2 + t*(1200*x[0]**2 - 400*x[1] + 2)
                dx1x2 = -400*t*x[0]
                dx1x3 = dx1x4 = 0
                dx2x2 = (1101*t)/5 + 1/(x[1] - 10)**2 + 1/(x[1] + 10)**2
                dx2x3 = 0
                dx2x4 = (99*t)/5
                dx3x3 = 1/(x[2] - 10)**2 + 1/(x[2] + 10)**2 + t*(1080*x[2]**2 - 360*x[3] + 2)
                dx3x4 = -360*t*x[2]
                dx4x4 = (1001*t)/5 + 1/(x[3] - 10)**2 + 1/(x[3] + 10)**2

                return np.array([
                    [dx1x1, dx1x2, dx1x3, dx1x4], [dx1x2, dx2x2, dx2x3, dx2x4],
                    [dx1x3, dx2x3, dx3x3, dx3x4], [dx1x4, dx2x4, dx3x4, dx4x4]
                ])

            return H

        def original(x):
            min_f = 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + 90*(x[2]**2 - x[3])**2 + (
                x[2] - 1)**2 + 10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8*(x[1] - 1)*(x[3] - 1)
            return x, min_f

        return set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original


def get_points(exe_num):
    if exe_num in [1, 3]:
        return [0, 0]

    elif exe_num == 2:
        return [1, 0.5, 3, 4]

    elif exe_num == 4:
        return None
 
    elif exe_num == 5:
        return np.array([2.0, 2.0, 2.0, 2.0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    points = get_points(args.exe_num)

    if args.exe_num == 1:
        x0, set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)
        print('Initial point: {}\n'.format(x0))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(
            set_f_t, set_g_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(
            set_f_t, set_g_t, set_H_t, original, points, eps, t=4, mu=1.2)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(
            set_f_t, set_g_t, original, points, eps, t=0.01, mu=2)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, set_H_t, set_get_f_t, set_get_g_t, original, points, eps, t=0.01, mu=2)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

    elif args.exe_num == 2:
        set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)
        print('Initial point: {}\n'.format(points))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(
            set_f_t, set_g_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(
            set_f_t, set_g_t, set_H_t, original, points, eps, t=1, mu=1.2)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(
            set_f_t, set_g_t, original, points, eps, t=2.5, mu=20)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, set_H_t, set_get_f_t, set_get_g_t, original, points, eps, t=2.5, mu=20)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

    elif args.exe_num == 3:
        x0, set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)
        print('Initial point: {}\n'.format(points))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(
            set_f_t, set_g_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(
            set_f_t, set_g_t, set_H_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(
            set_f_t, set_g_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, set_H_t, set_get_f_t, set_get_g_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

    elif args.exe_num == 4:
        x0, set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)
        print('Initial point: {}\n'.format(x0))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(
            set_f_t, set_g_t, original, x0, eps, t=1, mu=20)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(
            set_f_t, set_g_t, set_H_t, original, x0, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(
            set_f_t, set_g_t, original, x0, eps, t=1, mu=1.9)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, set_H_t, set_get_f_t, set_get_g_t, original, x0, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

    elif args.exe_num == 5:
        set_f_t, set_g_t, set_H_t, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)
        print('Initial point: {}\n'.format(points))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(
            set_f_t, set_g_t, original, points, eps, t=0.001, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(
            set_f_t, set_g_t, set_H_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(
            set_f_t, set_g_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, set_H_t, set_get_f_t, set_get_g_t, original, points, eps, t=1, mu=10)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))
