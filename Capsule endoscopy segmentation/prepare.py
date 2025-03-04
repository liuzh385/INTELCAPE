import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
from volumentations import *
from config import get_cfg_defaults
from models.utils import *
from functools import partial
import warnings
import itertools
import nibabel as nib
from tqdm import tqdm
import glob
from PIL import Image
import skimage.measure
# import skvideo.io
import multiprocessing

warnings.filterwarnings("ignore")


class Crohn2016(Dataset):

    def __init__(self, cfg, csv, train=True, pre=True):
        self.train = train
        self.csv = csv
        self.cfg = cfg
        self.df = pd.read_csv(csv)
        if pre is False:
            if self.train:
                self.flod = cfg.TRAIN.FOLD
                if self.flod == 0:
                    self.df = self.df[len(self.df) // 5:]
                elif self.flod == 1:
                    self.df = self.df.drop(range(len(self.df) // 5, len(self.df) // 5 * 2))
                elif self.flod == 2:
                    self.df = self.df.drop(range(len(self.df) // 5 * 2, len(self.df) // 5 * 3))
                elif self.flod == 3:
                    self.df = self.df.drop(range(len(self.df) // 5 * 3, len(self.df) // 5 * 4))
                elif self.flod == 4:
                    self.df = self.df[:len(self.df) // 5 * 4]
                else:
                    raise Exception("Flod ERROR")
                print("TRAIN SAMPLES")
                print(self.df)
            else:
                self.flod = cfg.VAL.FOLD
                if self.flod == 0:
                    self.df = self.df[: len(self.df) // 5]
                elif self.flod == 1:
                    self.df = self.df[len(self.df) // 5: len(self.df) // 5 * 2]
                elif self.flod == 2:
                    self.df = self.df[len(self.df) // 5 * 2: len(self.df) // 5 * 3]
                elif self.flod == 3:
                    self.df = self.df[len(self.df) // 5 * 3: len(self.df) // 5 * 4]
                elif self.flod == 4:
                    self.df = self.df[len(self.df) // 5 * 4:]
                else:
                    raise Exception("Flod ERROR")
                print("VAL SAMPLES")
                print(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        datum = self.df.iloc[index]
        sample_name = datum['Name']
        sample_name = sample_name.replace("-", "_")
        avi_name = os.path.join(self.cfg.DIRS.DATA, sample_name, sample_name[:-1] + '.avi')
        # tmp = cv2.VideoCapture(avi_name)
        # tmp.set(cv2.CAP_PROP_POS_FRAMES, 500)
        # a, b = tmp.read()
        # # print(b.shape, b[160, 160])
        # b[305:, :131, :] = [0, 0, 0]
        # b[:15, 260:, :] = [0, 0, 0]

        return avi_name, 50, datum['S_intestine_frame'], datum['L_intestine_frame']


def pre_moto():
    cfg = get_cfg_defaults()

    mode = 'train'

    dts1 = Crohn2016(cfg, csv=cfg.DIRS.DATA_CSV)
    batch_size = cfg.TRAIN.BATCH_SIZE
    dataloader1 = DataLoader(dts1, batch_size=batch_size,
                             shuffle=True, drop_last=False,
                             num_workers=cfg.SYSTEM.NUM_WORKERS)

    # dataloader1
    print(len(dataloader1))
    dataiter1 = iter(dataloader1)

    for i in range(len(dataloader1)):
        print(i, len(dataloader1))
        avi_name, _, _, _ = dataiter1.next()
        print(avi_name)
        tmp = cv2.VideoCapture(avi_name[0])
        # print(tmp.isOpened())
        frame_num = tmp.get(7)
        frame_num = int(frame_num)
        for i in range(frame_num):
            tmp.set(cv2.CAP_PROP_POS_FRAMES, i)
            a, b = tmp.read()
            frame_one = b[32:288, 32:288, :]
            frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
            np.save(f'{avi_name[0][:-4]}_{i}.npy', frame_one)


if __name__ == "__main__":
    param = []
    # root = "/GPUFS/sysu_gbli_1/zhaoxinkai/data/Crohn_avi"
    root = "/data2/zhaoxinkai/Crohn_avi"
    for fold in os.listdir(root):
        # if "莫灌申" in fold:
        name = os.path.join(root, fold, fold[:-1] + '.avi')
        tmp = cv2.VideoCapture(name)
        frame_num = tmp.get(7)
        # print(avi_name.split('/')[-2])
        len_file = len(os.listdir(os.path.join(root, name.split('/')[-2])))
        print(frame_num, len_file)
        if frame_num * 2 + 1 == len_file:
            print(name.split('/')[-2], "return")
        else:
            print(name.split('/')[-2], "!!!")
            param.append(name)

    print(param)

    def do(avi_name):
        print(avi_name)
        # videodata = skvideo.io.vread(avi_name)
        tmp = cv2.VideoCapture(avi_name)
        frame_num = tmp.get(7)
        # print(avi_name.split('/')[-2])
        len_file = len(os.listdir(os.path.join(root, avi_name.split('/')[-2])))
        print(frame_num, len_file)
        if frame_num * 2 + 1 == len_file:
            print(avi_name.split('/')[-2], "return")
            return

        for i in range(int(frame_num-1), 0, -1):
            # b = videodata[i]
            _, b = tmp.read()
            # frame_one = b[32:288, 32:288, :]
            frame_one = b

            # frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
            # np.save(f'{avi_name[:-4]}_{i}.npy', frame_one)
            print(f'{avi_name[:-4]}_{i}.jpg')
            # frame_one = cv2.cvtColor(frame_one, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{avi_name[:-4]}_{i}.jpg', frame_one)
            # print(f'{avi_name[:-4]}_{i}.jpg', '/', videodata.shape[0])
        print(avi_name.split('/')[-2], "finish")

    p = multiprocessing.Pool(8)
    b = p.map(do, param)
    p.close()
    p.join()
