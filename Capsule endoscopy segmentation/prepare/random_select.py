import cv2
import numpy as np
import os
import random
import shutil
import pandas as pd
from PIL import Image 
num_images = 1000
# root = "/mnt/minio/node77/liuzheng/RJ/Data/small_jpg"
# target = "/mnt/minio/node77/liuzheng/RJ/Data/random_jpg"
root = "/mnt/minio/node77/liuzheng/RJ/Data/random_jpg"
target = "/mnt/minio/node77/liuzheng/small_jpg"
tmp = os.listdir(root)
images = []
# for index in range(len(tmp)):
#     path = tmp[index]
#     path = os.path.join(root, path)
#     dir_path = os.listdir(path)
#     images = [os.path.join(path, f) for f in dir_path]
#     sample_images = random.sample(images, num_images)
#     for image in sample_images:
#         shutil.copy(image, target)
dir_path = os.listdir(root)
images = [os.path.join(root, f) for f in dir_path]
sample_images = random.sample(images, num_images)
for image in sample_images:
    shutil.copy(image, target)
