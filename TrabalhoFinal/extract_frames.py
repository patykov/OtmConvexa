import os

import cv2

vidcap = cv2.VideoCapture(
    'E:/ActionTraining/Dataset/kinetics/train/capoeira/__u8N7jWPsY_000387_000397.mp4')
file_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(file_path, 'capoeira')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

success, image = vidcap.read()
count = 0
while success:
    success, image = vidcap.read()
    cv2.imwrite('{}/{}.jpg'.format(output_dir, count), image)
    # print('Read a new frame: ', success)
    count += 1
