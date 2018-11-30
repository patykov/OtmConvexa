import numpy as np

import sys
sys.path.append('../')  # Fix it later, import from other dir not working
from lista03 import functions as f3
from lista02 import functions as f2


def iterative_conjugate_gradient(f, g, H, p, eps):
    # initial point
    x, fx, k = f3.conjugate_gradient(f, g, H, p, eps)
    t = np.arange(0.2, 10000, 0.1)
    all_k = k
    for ti in t:
        x, fx, k = f3.conjugate_gradient(f, g, H, x, eps, t=ti)
        all_k += k

    return x, fx, all_k