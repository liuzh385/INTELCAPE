import os
import numpy as np
import pandas as pd
import cv2
import warnings
from tqdm import tqdm
from PIL import Image
import multiprocessing
import random

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    param = []
    csv_file = "/mnt/minio/node77/liuzheng/RJ/Data/csv/RJ_select.csv" # csv路径
    df = pd.read_csv(csv_file)
    root = "/mnt/minio/node77/liuzheng/RJ/Data/RJ_mp4" # 视频根路径
    save_root = "/mnt/minio/node77/liuzheng/RJ/Data/small_jpg" # 保存地址
    frame = {}
    u_name = []
    print(df)
    for index in range(len(df)):
        datum = df.iloc[index]
        sample_name = datum['Name']
        s_frame = datum['S_intestine_frame']
        l_frame = datum['L_intestine_frame']
        frame[datum['Name']] = (s_frame, l_frame)
        u_name.append(datum['Name'])
        print(datum['Name'], (s_frame, l_frame))

    os.makedirs(save_root, exist_ok=True)
    for f in os.listdir(root):
        if f == 'img':
            continue
        if f not in u_name:
            continue
        name = os.path.join(root, f, f+'.mp4')
        # os.makedirs(os.path.join(save_root, name.split('/')[-2]), exist_ok=True)
        frame_num = frame[f][1] - frame[f][0]
        len_file = 0
        save_name = f
        if save_name in os.listdir(save_root):
            len_file = len(os.listdir(os.path.join(save_root, save_name)))
        if frame_num <= len_file:
            print(name.split('/')[-2], "return")
        else:
            print(name, os.path.join(save_root, name.split('/')[-2]))
            print(frame_num, len_file)
            print(name.split('/')[-2], "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            param.append(name)

    print(param)
    # print("before do")
    def do(avi_name):
        print(avi_name)
        if os.path.exists(avi_name):  
            save_name = avi_name.split('/')[-1][:-4] # 去掉.avi 只留下视频的名字
            print(save_name)
            tmp = cv2.VideoCapture(avi_name)
            whole_frame_num = tmp.get(7)
            frame_num = frame[save_name][1] - frame[save_name][0]
            len_file = 0
            if save_name in os.listdir(save_root):
                len_file = len(os.listdir(os.path.join(save_root, save_name)))
            print(save_name, whole_frame_num, frame_num, len_file, frame[save_name][1], '->', frame[save_name][0])
            os.makedirs(os.path.join(save_root, save_name), exist_ok=True)
            if frame_num == len_file:
                print(save_name, "return")
                return

            len_file -= 100
            len_file = max(0, len_file)
            print(os.path.join(save_root, save_name))
            for j in range(5000):
                i = random.randint(int(frame[save_name][0]), int(frame[save_name][1]-len_file-1))
                tmp.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, b = tmp.read()
                # print("in for")
                frame_one = b
                cv2.imwrite(os.path.join(save_root, save_name, f'{i}.jpg'), frame_one)
            print(avi_name.split('/')[-2], save_name, "finish")
            
    p = multiprocessing.Pool(8)
    b = p.map(do, param)
    p.close()
    p.join()
    # for name in param:
    #     do(name)