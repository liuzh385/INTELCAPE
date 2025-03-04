import os
import pandas as pd
import cv2
from models.utils import *
import warnings
import multiprocessing

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    param = []
    csv = "./data/Crohn_all_9.csv"
    root = "/home/zhaoxinkai/data/Crohn_avi_2020"  # path of video
    save_root = "/home/zhaoxinkai/data/Crohn_jpg_2020"  # path of jpg
    df = pd.read_csv(csv)
    frame = {}
    u_name = {}
    print(df)
    for index in range(len(df)):
        datum = df.iloc[index]
        if len(str(datum['Name'])) < 5:
            continue
        if datum['CN_Name'] not in os.listdir(root):
            continue
        sample_name = datum['CN_Name']
        s_frame = datum['S_intestine_frame']
        l_frame = datum['L_intestine_frame']
        sample_name = sample_name.replace("-", "_")
        frame[datum['Name']] = (s_frame, l_frame)
        u_name[sample_name] = datum['Name']
        print(datum['Name'], (s_frame, l_frame))

    os.makedirs(save_root, exist_ok=True)
    for f in os.listdir(root):
        if f == 'img':
            continue
        if f not in u_name:
            continue
        fold = u_name[f]
        if fold not in frame:
            continue
        name = os.path.join(root, f, f+'.avi')
        # os.makedirs(os.path.join(save_root, name.split('/')[-2]), exist_ok=True)
        frame_num = frame[fold][1] - frame[fold][0]
        len_file = 0
        save_name = u_name[name.split('/')[-2]]
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

    def do(avi_name):
        print(avi_name)
        save_name = avi_name.split('/')[-1][:-4]
        save_name = u_name[save_name]
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
        for i in range(int(frame[save_name][1]-len_file-1), int(frame[save_name][0])-1, -1):
            tmp.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, b = tmp.read()
            frame_one = b

            cv2.imwrite(os.path.join(save_root, save_name, f'{i}.jpg'), frame_one)
        print(avi_name.split('/')[-2], save_name, "finish")
        # os.remove(avi_name)

    p = multiprocessing.Pool(32)
    b = p.map(do, param)
    p.close()
    p.join()
