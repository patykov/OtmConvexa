import cv2
import os

vidcap = cv2.VideoCapture('dataset/person01_walking_d1_uncomp.avi')
output_dir = 'dataset/walking1'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    cv2.imwrite('dataset/walking1/{}.jpg'.format(count), image)
    # print('Read a new frame: ', success)
    count += 1
