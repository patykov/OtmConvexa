from __future__ import division, print_function
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pcp2 import pcp
from r_pca import R_pca


def bitmap_to_mat(bitmap_seq):
    """from blog.shriphani.com"""
    matrix = []
    shape = None
    for bitmap_file in bitmap_seq:
        img = Image.open(bitmap_file).convert("L")
        if shape is None:
            shape = img.size
        assert img.size == shape
        img = np.array(img.getdata())
        matrix.append(img)
    return np.array(matrix), shape[::-1]


def do_plot(ax, img, shape):
    ax.cla()
    ax.imshow(img.reshape(shape), cmap="gray", interpolation="nearest")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


if __name__ == "__main__":

    if "--test" in sys.argv:
        M = (10*np.ones((10, 10))) + (-5 * np.eye(10))
        L, S, svd = pcp(M, verbose=True, svd_method="exact")
        assert np.allclose(M, L + S), "Failed"
        print("passed")
        sys.exit(0)

    files_list = glob.glob("dataset/walking1/*.jpg")[:60]
    sorted_list = sorted(files_list, key=lambda x: int(x.split(os.sep)[-1].replace('.jpg', '')))
    M, shape = bitmap_to_mat(sorted_list)
    print(M.shape)

    # L, S, (u, s, v) = pcp(M, maxiter=50, verbose=True, svd_method="exact")

    # rpca = R_pca(M)
    # L, S = rpca.fit(max_iter=10000, iter_print=100)
    # rpca.plot_fit()
    # plt.show()

    L, S = pcp(M)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.subplots_adjust(left=0, right=1, hspace=0, wspace=0.01)
    for i in range(min(len(M), 500)):
        do_plot(axes[0], M[i], shape)
        axes[0].set_title("raw")
        do_plot(axes[1], L[i], shape)
        axes[1].set_title("low rank")
        do_plot(axes[2], S[i], shape)
        axes[2].set_title("sparse")
        fig.savefig("results/{0:05d}.png".format(i))
