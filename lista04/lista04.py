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

        def H(z):
            logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
            logs_dev = np.array([[F[i, 0] / logs_divs[i] for i in range(4)],
                                 [F[i, 1] / logs_divs[i] for i in range(4)]])
            aux = logs_dev**2
            dz1z1 = sum(aux[0])
            dz2z2 = sum(aux[1])
            dz1z2 = sum([(F[i, 0] * F[i, 1]) / logs_divs[i]**2 for i in range(4)])

            return np.array([[dz1z1, dz1z2], [dz1z2, dz2z2]])

        def original(z):
            min_x = np.dot(F, z) + x0
            min_fx = np.dot(c, min_x)

            return min_x, min_fx

        return x0, set_f_t, set_g_t, H, set_get_f_t, set_get_g_t, original

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
                dx2 = 2*t*(x[0] - x[2]) - logs_div[1] * (-(5/4.0)*x[2] - (3/4.0)*x[3] + 11/2.0)
                dx3 = 2*t*(x[1] - x[3]) - logs_div[1] * (-(3/4.0)*x[2] - (5/4.0)*x[3] + 13/2.0)

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

        def H(z):
            logs_div = [1/c1(x[:2]), 1/c2(x[2:])]
            dc1x1 = -0.5*x[0] + 0.5
            dc1x2 = -2*x[1]
            dc2x3 = -(5/4.0)*x[2] - (3/4.0)*x[3] + 11/2.0
            dc2x4 = -(3/4.0)*x[2] - (5/4.0)*x[3] + 13/2.0

            dx1x1 = 2*t + dc1x1**2 + 0.5*logs_div[0]
            dx1x2 = dc1x1*dc1x2
            dx1x3 = -2*t
            dx1x4 = 0

            dx2x2 = 2*t + dc1x2**2 + 2*logs_div[0]
            dx2x3 = 0
            dx2x4 = -2*t

            dx3x3 = 2*t + dc2x3**2 + (5/4.0)*logs_div[1]
            dx3x4 = dc2x3*dc2x4 + (3/4.0)*logs_div[1]

            dx4x4 = 2*t + dc2x4**2 + (5/4.0)*logs_div[1]

            return np.array([
                [dx1x1, dx1x2, dx1x3, dx1x4], [dx1x2, dx2x2, dx2x3, dx2x4],
                [dx1x3, dx2x3, dx3x3, dx3x4], [dx1x4, dx2x4, dx3x4, dx4x4]
                ])

        def original(x):
            x = np.array(x)
            return x, np.linalg.norm(x[:2] - x[2:])

        return set_f_t, set_g_t, H, set_get_f_t, set_get_g_t, original

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
                    (np.dot(F, z) + x0), C), np.dot(F, z) + x0) + np.dot(np.dot(F, z) + x0, d)
                I_ = [restricted_log(np.dot(F[i, :], z) + x0[i]) for i in range(3)]

                return t*0.5*f0 - sum(I_)

            return f

        def set_get_f_t(t):
            def get_f(z0, d):
                def f(alpha):
                    z = z0 + alpha * d
                    f0 = np.dot(0.5*np.dot(
                        (np.dot(F, z) + x0), C), np.dot(F, z) + x0) + np.dot(np.dot(F, z) + x0, d)
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
                df03 = [x0[i] + F[i, 0]*z[i] + F[i, 1]*z[1] for i in range(3)]
                df04 = [sum([
                    df03[i]*(C[0, i]*F_conj[0, j] + C[1, i]*F_conj[1, j] + C[2, i]*F_conj[2, j]
                    )/2.0 - F[i, j]/df03[i] for i in range(3)])
                        for j in range(2)]

                return np.array([np.array(df01) + np.array(df02) + np.array(df03) + np.array(df04)])

            return g

        def set_get_g_t(t):
            def get_g(z0, d):
                def g(alpha):
                    z = z0 + alpha * d
                    z_conj = np.conj(z)
                    F_conj = np.conj(F)
                    x0_conj = np.conj(x0)

                    df01 = [
                        sum([F[i, j]*(
                            C[i, 0]*(x0_conj[i] + F_conj[i, 0]*z_conj[0] + F_conj[i, 1]*z_conj[1]))/2.0
                            for i in range(3)]) for j in range(2)]
                    df02 = [sum([d[i]*F_conj[i, j] for i in range(3)])
                            for j in range(2)]
                    df03 = [x0[i] + F[i, 0]*z[i] + F[i, 1]*z[1] for i in range(3)]
                    df04 = [sum([
                        df03[i]*(C[0, i]*F_conj[0, j] + C[1, i]*F_conj[1, j] + C[2, i]*F_conj[2, j]
                        )/2.0 - F[i, j]/df03[i] for i in range(3)])
                            for j in range(2)]

                    return np.array([np.array(df01) + np.array(df02) + np.array(df03) + np.array(df04)])
                return g

            return get_g

        def H(z):

aaa = (H11*conj(F11))/2 + (H21*conj(F21))/2 + (H31*conj(F31))/2
ddd = (H11*conj(F12))/2 + (H21*conj(F22))/2 + (H31*conj(F32))/2

bbb = (H12*conj(F11))/2 + (H22*conj(F21))/2 + (H32*conj(F31))/2
eee = (H12*conj(F12))/2 + (H22*conj(F22))/2 + (H32*conj(F32))/2

ccc = (H13*conj(F11))/2 + (H23*conj(F21))/2 + (H33*conj(F31))/2
fff = (H13*conj(F12))/2 + (H23*conj(F22))/2 + (H33*conj(F32))/2

xxx = x01 + F11*z1 + F12*z2
yyy = x02 + F21*z1 + F22*z2
zzz = x03 + F31*z1 + F32*z2

dz1z1 = (2*F11*(aaa) + 2*F21*(bbb) + 2*F31*(ccc)) + F11^2/(xxx)^2 + F21^2/(yyy)^2 + F31^2/(zzz)^2
dz1z2 = (F12*(aaa) + F11*(ddd) + F22*(bbb) + F21*(eee) + F32*(ccc) + F31*(fff)) + (F11*F12)/(xxx)^2 + (F21*F22)/(yyy)^2 + (F31*F32)/(zzz)^2
dz2z2 = (2*F12*(ddd) + 2*F22*(eee) + 2*F32*(fff)) + F12^2/(xxx)^2 + F22^2/(yyy)^2 + F32^2/(zzz)^2

            F_conj = np.conj(F)
            der1 = [
                sum([(C[i, j]*(F_conj[1, 0]))/2.0  for i in range(3)]) 
                for j in range(3)]

            logs_divs = [(x0[i] + F[i, 0] * z[0] + F[i, 1] * z[1]) for i in range(4)]
            logs_dev = np.array([[F[i, 0] / logs_divs[i] for i in range(4)],
                                 [F[i, 1] / logs_divs[i] for i in range(4)]])
            aux = logs_dev**2
            dz1z1 = sum(aux[0])
            dz2z2 = sum(aux[1])
            dz1z2 = sum([(F[i, 0] * F[i, 1]) / logs_divs[i]**2 for i in range(4)])

            return t*np.array([[dz1z1, dz1z2], [dz1z2, dz2z2]])

        def original(z):
            min_x = np.dot(F, z) + x0
            min_fx = 0.5*np.dot(np.dot(min_x, C), min_x) + np.dot(min_x, d)

            return min_x, min_fx

        return x0, set_f_t, set_g_t, H, set_get_f_t, set_get_g_t, original


def get_points(exe_num):
    if exe_num in [1, 3]:
        return [0, 0]

    elif exe_num == 2:
        # return [2.0447, 0.8527, 2.5449, 2.4857]
        return [0, 0, 2, 4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exe_num', help='Number of the exercise.', required=True, type=float)
    args = parser.parse_args()

    eps = 10**(-6)
    points = get_points(args.exe_num)

    if args.exe_num == 1:
        x0, set_f_t, set_g_t, H, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)
        print('Initial point: {}\n'.format(x0))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(set_f_t, set_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(set_f_t, set_g_t, H, original, points,
                                                              eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(set_f_t, set_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, H, set_get_f_t, set_get_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

    elif args.exe_num == 2:
        set_f_t, set_g_t, H, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)

        print('Initial point: {}\n'.format(points))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(set_f_t, set_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(set_f_t, set_g_t, H, original, points,
                                                              eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(set_f_t, set_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, H, set_get_f_t, set_get_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

    elif args.exe_num == 3:
        x0, set_f_t, set_g_t, H, set_get_f_t, set_get_g_t, original = get_defined_functions(
            args.exe_num)

        f = set_f_t(1)
        print(f(points))

        x, fx = original(points)
        print(fx)

        g = set_g_t(1)
        print(g(points))
        raise Exception

        print('Initial point: {}\n'.format(points))

        print('Steepest Descent')
        t = time.time()
        x, fx, k, it = functions.iterative_steepest_descent(set_f_t, set_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Conjugate Gradiente')
        t = time.time()
        x, fx, k, it = functions.iterative_conjugate_gradient(set_f_t, set_g_t, H, original, points,
                                                              eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Quasi-Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_quasi_newton(set_f_t, set_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))

        print('Newton')
        t = time.time()
        x, fx, k, it = functions.iterative_newton(
            set_g_t, H, set_get_f_t, set_get_g_t, original, points, eps)
        delta_t = (time.time() - t) * 1000
        print('x: {}, fx: {:.5f}, num_iter: {}, calls: {}, time: {:.5f} ms\n'.format(
            x, fx, k, it, delta_t))
