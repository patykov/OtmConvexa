

# def backtracting_line_search(f, delta_x, ):

def dichotomos_search(f, xl, xu, eps, max_iter=50):
    num_iter = 0

    while (xu - xl > eps) and (num_iter < max_iter):
        x = (xu + xl)/2
        xa = x - eps/2
        xb = x + eps/2

        num_iter += 1
        if f(xa) < f(xb):
            xu = xb
        else:
            xl = xa

        print('xl: {:.20f}, xu:{:.20f}, xa:{:.20f}, f(xa):{:.20f}, xb: {:.20f}, f(xb):{:.20f}\
        '.format(
            xl, xu, xa, f(xa), xb, f(xb)))
        print('x: {:.20f}, f(x): {:.20f}, num_iter: {}, xu-xl:{:.20f}, xu-xl/2:{:.20f}\n'.format(
            x, f(x), num_iter, xu - xl, (xu - xl)/2))
