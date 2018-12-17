from __future__ import division, print_function

import glob
import os
import time

import numpy as np
from PIL import Image
from skimage import filters

from pcp import my_pcp


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


if __name__ == "__main__":
    dataset = 2

    if dataset == 1:  # Wallflower
        frames_dir = "C:/Users/v-pakova/source/repos/OtmConvexa/TrabalhoFinal/dataset/" \
                    "MovedObject"
        files_list = glob.glob("{}/*.bmp".format(frames_dir))
        sorted_list = sorted(files_list, key=lambda x: int(
            os.path.basename(x).replace('.bmp', '').replace('b', '')))[1380:1440]

    elif dataset == 2:
        frames_dir = "C:/Users/v-pakova/source/repos/OtmConvexa/TrabalhoFinal/dataset/" \
                    "winterDriveway/input"
        files_list = glob.glob("{}/*.jpg".format(frames_dir))
        sorted_list = sorted(files_list, key=lambda x: int(
            os.path.basename(x).replace('.jpg', '').replace('in', '')))[1800:1860]

    elif dataset == 3:  # ucf101
        frames_dir = "C:/Users/v-pakova/source/repos/OtmConvexa/TrabalhoFinal/dataset/" \
                    "capoeira"
        files_list = glob.glob("{}/*.jpg".format(frames_dir))
        sorted_list = sorted(files_list, key=lambda x: int(
            os.path.basename(x).replace('.jpg', '')))[:60]

    M, shape = bitmap_to_mat(sorted_list)
    print(M.shape)

    t = time.time()

    L, S = my_pcp(M, shape, connected=False)

    delta_t = (time.time() - t) * 1000
    print("Total time: {:.5f} ms".format(delta_t))

    threshold = filters.threshold_otsu(S)
    St = S < threshold
    St = np.array(St, dtype=np.float32)*255

    output_dir = 'data{}-result-pcp'.format(dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(min(len(M), 500)):
        data = [M, L, S, St]
        text = ['original', 'low_rank', 'sparse', 'sparse_t']
        for d, t in zip(data, text):
            img = Image.fromarray(d[i].reshape(shape))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save("{}/{}_{}.png".format(output_dir, i, t))
