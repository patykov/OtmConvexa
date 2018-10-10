

# def backtracting_line_search(f, delta_x, ):

def dichotomos_search(f, xl, xu, uncertainty_range, max_iter=20):
    num_iter = 0
    eps = uncertainty_range/10

    while (xu - xl > uncertainty_range):
        x = (xu + xl)/2
        xa = x - eps/2
        xb = x + eps/2

        num_iter += 1
        if f(xa) < f(xb):
            xu = xb
        else:
            xl = xa

        if num_iter >= max_iter:
            break

    return x, f(x), num_iter


def fibonacci_search(f, xl, xu, uncertainty_range, max_iter=20):
    Il = (xu - xl)
    Fn = Il/(uncertainty_range)

    F = [1, 1]
    while F[-1] < Fn:
        F.append(sum(F[-2:])*1.0)
    n = len(F)

    Il *= (F[n-2]/F[n-1])
    xa = xu - Il
    xb = xl + Il
    fa = f(xa)
    fb = f(xb)

    for k in range(1, n-1):
        Il *= (F[n-k-2]/F[n-k-1])

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
