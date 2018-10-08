

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
