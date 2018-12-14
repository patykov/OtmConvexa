from __future__ import division, print_function

import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters

from ialm import inexact_augmented_lagrange_multiplier
from pcp import pcp
from r_pca import R_pca
from rpca import rpca
from rpcaADMM import rpcaADMM


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

    frames_dir = "C:/Users/v-pakova/source/repos/OtmConvexa/TrabalhoFinal/capoeira"

    files_list = glob.glob("{}/*.jpg".format(frames_dir))
    sorted_list = sorted(files_list, key=lambda x: int(
        os.path.basename(x).replace('.jpg', '')))[:30]
    M, shape = bitmap_to_mat(sorted_list)
    print(M.shape)

    t = time.time()
    A, E = inexact_augmented_lagrange_multiplier(M/255.)
    L = np.array([np.reshape(Ai, shape) for Ai in A])*255.
    S = np.array([np.reshape(Ei, shape) for Ei in E])*255.

    # L, S = rpca(M)

    # L, S, (u, s, v) = pcp(M, maxiter=50, verbose=True, svd_method="sparse")

    # rpca = R_pca(M)
    # L, S = rpca.fit(max_iter=100, iter_print=100)
    # rpca.plot_fit()
    # plt.show()

    # h = rpcaADMM(M)
    # L = h['X3_admm']
    # S = h['X2_admm']

    delta_t = (time.time() - t) * 1000
    print("Total time: {:.5f} ms".format(delta_t))

    # threshold = filters.threshold_isodata(S)
    # S = S > threshold

    output_dir = 'resultsialm'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.subplots_adjust(left=0, right=1, hspace=0, wspace=0.01)
    for i in range(min(len(M), 500)):
        do_plot(axes[0], M[i], shape)
        axes[0].set_title("raw")
        do_plot(axes[1], L[i], shape)
        axes[1].set_title("low rank")
        do_plot(axes[2], S[i], shape)
        axes[2].set_title("sparse")
        fig.savefig("{}/{}.png".format(output_dir, i))
