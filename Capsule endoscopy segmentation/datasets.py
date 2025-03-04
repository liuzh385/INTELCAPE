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
import multiprocessing
warnings.filterwarnings("ignore")

from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Crohn2016_jpg(Dataset):

    def __init__(self, cfg, csv, mode="train"):
        self.mode = mode
        self.csv = csv
        self.cfg = cfg
        self.df = pd.read_csv(csv)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.step = cfg.DATA.STEP
        if self.mode == "train":
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
            self.df = self.df[:-5]
            print("TRAIN SAMPLES")
        elif self.mode == "val":
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
            self.df = self.df[-5:]
            print("VAL SAMPLES")
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
            print("TEST SAMPLES")
        # self.df = self.df + self.df
        print(self.df)

    def __len__(self):
        return len(self.df) * self.batch_size

    def __getitem__(self, index):
        index_moto = index
        index = index % len(self.df)
        datum = self.df.iloc[index]
        sample_name = datum['Name']
        sample_name = sample_name.replace("-", "_")
        avi_name = os.path.join(self.cfg.DIRS.DATA, sample_name, sample_name[:-1] + '.avi')

        # frame_num = len(os.listdir(os.path.join(self.cfg.DIRS.DATA, sample_name))) - 1

        tmp = cv2.VideoCapture(avi_name)
        # print(avi_name, tmp.isOpened())
        frame_num = tmp.get(7)
        # frame_num = 50000
        # frames = None
        # labels = np.array([])

        stomach_frame = 50

        if self.cfg.DATA.NUM == 1:
            index = random.randint(stomach_frame, frame_num - 1)
            # index = torch.randint(stomach_frame, int(frame_num - 1), (1,)).item()
            frame = Image.open(avi_name[:-4] + f"_{index}.jpg")
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # print(flag)
            frame[305:, :131, :] = [0, 0, 0]
            frame[:15, 260:, :] = [0, 0, 0]
            frame = frame.transpose(2, 0, 1)
        else:
            if self.mode == "train":
                # index = random.randint(stomach_frame + (self.cfg.DATA.NUM // 2 * self.step),
                #                        frame_num - (self.cfg.DATA.NUM * self.step))
                index = torch.randint(stomach_frame + (self.cfg.DATA.NUM // 2 * self.step),
                                      int(frame_num - (self.cfg.DATA.NUM * self.step)), (1,)).item()
            else:
                a = frame_num
                b = float((index_moto % self.batch_size) / self.batch_size)
                # print(index_moto, self.batch_size)
                # print(a, b, a * b)
                index = a * b
                index = int(index)
                index = max(index, stomach_frame + (self.cfg.DATA.NUM // 2 * self.step))
                index = min(index, frame_num - (self.cfg.DATA.NUM * self.step))
                # print('3----', index)
            for j in range(self.cfg.DATA.NUM):
                # frame_one = Image.open(avi_name[:-4] +
                #                        f"_{index - (self.cfg.DATA.NUM // 2 * self.step) + (j * self.step)}.jpg")
                # frame_one = np.array(frame_one)
                # frame_one = cv2.cvtColor(frame_one, cv2.COLOR_RGB2BGR)

                frame_one = cv2.imread(avi_name[:-4] +
                                       f"_{index - (self.cfg.DATA.NUM // 2 * self.step) + (j * self.step)}.jpg")
                frame_one[305:, :131, :] = [0, 0, 0]
                frame_one[:15, 260:, :] = [0, 0, 0]
                frame_one = frame_one[np.newaxis, :]
                # cv2.imwrite("temp.jpg", frame_one[0])
                if j == 0:
                    frame = frame_one
                else:
                    frame = np.concatenate((frame, frame_one), axis=0)
            # print(frame.shape)
            # frame = frame.transpose(0, 3, 1, 2)
            # frame[305:, :131, :] = [0] * self.cfg.DATA.NUM * 3
            # frame[:15, 260:, :] = [0] * self.cfg.DATA.NUM * 3

            # frame = frame.transpose(3, 0, 1, 2)
            frame = frame.transpose(0, 3, 1, 2)

        if index < datum['S_intestine_frame']:
            labels = 0
        elif index > datum['L_intestine_frame']:
            labels = 2
        else:
            labels = 1

        # get label
        # 0: stomach, 1: small intestine, 2: large intestine
        # print(frames.dtype, labels.dtype)
        return torch.from_numpy(frame).float(), torch.tensor(labels).long()

        # return avi_name, 50, datum['S_intestine_frame'], datum['L_intestine_frame']


class Crohn15to23(Dataset):

    def __init__(self, cfg, csv, mode=None):
        print('====>', mode)
        self.mode = mode
        self.csv = csv
        self.cfg = cfg
        self.df = pd.read_csv(csv)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.step = cfg.DATA.STEP
        if self.mode == "train":
            self.flod = cfg.TRAIN.FOLD
            if self.flod == 0:
                self.df = self.df[40:]
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
            self.df = self.df[:-5]
            print("TRAIN SAMPLES")
        elif self.mode == "val":
            self.flod = cfg.VAL.FOLD
            if self.flod == 0:
                self.df = self.df[40:]
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
            self.df = self.df[-5:]
            print("VAL SAMPLES")
        else:
            self.flod = cfg.TEST.FOLD
            if self.flod == 0:
                self.df = self.df[: 40]
            elif self.flod == 1:
                self.df = self.df[len(self.df) // 5: len(self.df) // 5 * 2]
            elif self.flod == 2:
                self.df = self.df[len(self.df) // 5 * 2: len(self.df) // 5 * 3]
            elif self.flod == 3:
                self.df = self.df[len(self.df) // 5 * 3: len(self.df) // 5 * 4]
            elif self.flod == 4:
                self.df = self.df[len(self.df) // 5 * 4:]
            else:
                print(self.flod)
                raise Exception("Flod ERROR")
            print("TEST SAMPLES")
        print(self.df)

    def __len__(self):
        return len(self.df) * self.batch_size

    def __getitem__(self, index):
        index_moto = index
        index = index % len(self.df)
        index = int(index)
        datum = self.df.iloc[index]
        sample_name = datum['Name']
        stomach_frame, s_intestine_frame, l_intestine_frame = datum['Stomach_frame'], datum['S_intestine_frame'], datum['L_intestine_frame']
        stomach_frame = int(stomach_frame)
        s_intestine_frame = int(s_intestine_frame)
        l_intestine_frame = int(l_intestine_frame)
        # sample_name = sample_name.replace("-", "_")
        # avi_name = os.path.join(self.cfg.DIRS.DATA, sample_name, sample_name[:-1] + '.avi')
        avi_name = os.path.join(self.cfg.DIRS.DATA, sample_name, sample_name + '.mp4')
        # get frame
        tmp = cv2.VideoCapture(avi_name)
        # print(avi_name[0], tmp.isOpened())
        frame_num = int(tmp.get(7))
        if self.cfg.DATA.NUM == 1:
            index = random.randint(stomach_frame, frame_num - 1)
            tmp.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = tmp.read()
            if self.cfg.DATA.CROP == 128:
                frame = frame[32:288, 32:288, :]
                frame = skimage.measure.block_reduce(frame, (2, 2, 1), np.max)
            else:
                # frame[305:, :131, :] = [0, 0, 0]
                # frame[:15, 260:, :] = [0, 0, 0]
                # RJ
                frame[:80, :100, :] = [0, 0, 0]
                frame[:30, 100:, :] = [0, 0, 0]
                frame[540:, :200, :] = [0, 0, 0]
                # SN
                # frame[:12 , 190:, :] = 0
            frame = frame.transpose(2, 0, 1)
            # frames = frame[np.newaxis, :]
        else:
            if self.mode == "train":
                # index = random.randint(stomach_frame + (self.cfg.DATA.NUM // 2 * self.step),
                #                        frame_num - (self.cfg.DATA.NUM * self.step))
                # print("Step:", self.step)
                # print("frame_num:", frame_num)
                # print("stomach_frame:", stomach_frame)
                # print("DATA.NUM:", self.cfg.DATA.NUM)
                if frame_num == 0:
                    print("miss:", avi_name)
                index = torch.randint(stomach_frame + (self.cfg.DATA.NUM // 2 * self.step),
                                      int(frame_num - (self.cfg.DATA.NUM * self.step)), (1,)).item()
            else:
                a = frame_num
                b = float((index_moto % self.batch_size) / self.batch_size)
                # print(index_moto, self.batch_size)
                # print(a, b, a * b)
                index = a * b
                index = int(index)
                index = max(index, stomach_frame + (self.cfg.DATA.NUM // 2 * self.step))
                index = min(index, frame_num - (self.cfg.DATA.NUM * self.step))
                # print('3----', index)
            for j in range(self.cfg.DATA.NUM):
                tmp.set(cv2.CAP_PROP_POS_FRAMES, int(index - (self.cfg.DATA.NUM // 2 * self.step) + (j * self.step)))
                flag, frame_one = tmp.read()
                # print(flag)
                if flag == False:
                    # print(index)
                    print(avi_name, int(index - (self.cfg.DATA.NUM // 2 * self.step) + (j * self.step)))
                if self.cfg.DATA.CROP == 128:
                    frame_one = frame_one[32:288, 32:288, :]
                    frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
                else:
                    # frame_one[305:, :131, :] = [0, 0, 0]
                    # frame_one[:15, 260:, :] = [0, 0, 0]
                    
                    frame_one[:80, :100, :] = [0, 0, 0]
                    frame_one[:30, 100:, :] = [0, 0, 0]
                    frame_one[540:, :200, :] = [0, 0, 0]
                    
                    # frame_one[:12 , 190:, :] = 0
                frame_one = frame_one[np.newaxis, :]
                # cv2.imwrite("temp.jpg", frame_one[0])
                if j == 0:
                    frame = frame_one
                else:
                    frame = np.concatenate((frame, frame_one), axis=0)
            # print(frame.shape)
            # frame = frame.transpose(0, 3, 1, 2)
            # frame[305:, :131, :] = [0] * self.cfg.DATA.NUM * 3
            # frame[:15, 260:, :] = [0] * self.cfg.DATA.NUM * 3

            # frame = frame.transpose(3, 0, 1, 2)
            frame = frame.transpose(0, 3, 1, 2)

        if index < s_intestine_frame:
            labels = 0
        elif index > l_intestine_frame:
            labels = 2
        else:
            labels = 1

        # get label
        # 0: stomach, 1: small intestine, 2: large intestine
        # print(frames.shape, labels.shape)
        return torch.from_numpy(frame).float(), torch.tensor(labels).long()

    
class Crohn2016(Dataset):

    def __init__(self, cfg, csv, mode=None):
        print('====>', mode)
        self.mode = mode
        self.csv = csv
        self.cfg = cfg
        self.df = pd.read_csv(csv)
        if self.mode == "train":
            self.flod = cfg.TRAIN.FOLD
            if self.flod == 0:
                self.df = self.df[:]
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
            self.df = self.df[:-5]
            print("TRAIN SAMPLES")
        elif self.mode == "val":
            self.flod = cfg.VAL.FOLD
            if self.flod == 0:
                self.df = self.df[:]
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
            self.df = self.df[-5:]
            print("VAL SAMPLES")
        else:
            self.flod = cfg.TEST.FOLD
            if self.flod == 0:
                self.df = self.df[:]
            elif self.flod == 1:
                self.df = self.df[len(self.df) // 5: len(self.df) // 5 * 2]
            elif self.flod == 2:
                self.df = self.df[len(self.df) // 5 * 2: len(self.df) // 5 * 3]
            elif self.flod == 3:
                self.df = self.df[len(self.df) // 5 * 3: len(self.df) // 5 * 4]
            elif self.flod == 4:
                self.df = self.df[len(self.df) // 5 * 4:]
            else:
                print(self.flod)
                raise Exception("Flod ERROR")
            print("TEST SAMPLES")
        print(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        datum = self.df.iloc[index]
        sample_name = datum['Name']
        # sample_name = sample_name.replace("-", "_")
        avi_name = os.path.join(self.cfg.DIRS.DATA, sample_name, sample_name[:] + '.avi')
        # avi_name = os.path.join(self.cfg.DIRS.DATA, sample_name, sample_name + '.mp4')

        # return avi_name, 50, datum['S_intestine_frame'], datum['L_intestine_frame']
        return avi_name, datum['Stomach_frame'], datum['S_intestine_frame'], datum['L_intestine_frame']

# TODO : STEP
class Crohn2016Npy(Dataset):

    def __init__(self, cfg, csv, train=True):
        self.train = train
        self.csv = csv
        self.cfg = cfg
        self.df = pd.read_csv(csv)
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
        temp = []
        for index in range(len(self.df)):
            datum = self.df.iloc[index]
            print('=====>', datum)
            sample_name = datum['Name']
            s_frame = datum['S_intestine_frame']
            l_frame = datum['L_intestine_frame']
            sample_name = sample_name.replace("-", "_")
            print(sample_name, index, len(self.df))
            avi_name = os.path.join(self.cfg.DIRS.DATA, sample_name, sample_name[:-1] + '.avi')
            tmp = cv2.VideoCapture(avi_name)
            frame_num = tmp.get(7)
            frame_num = int(frame_num)
            for i in range(frame_num):
                if i < s_frame:
                    label = 0
                elif i > l_frame:
                    label = 2
                else:
                    label = 1
                temp.append((f'{avi_name[:-4]}_{i}.npy', label, 50, frame_num, i))
        random.shuffle(temp)
        self.df2 = temp

    def __len__(self):
        return len(self.df2)

    def __getitem__(self, index):
        frame_name, label, start, end, i = self.df2[index]
        if self.cfg.DATA.NUM == 1:
            frame = np.load(frame_name)
            frame = frame.transpose(2, 0, 1)
        else:
            index = i
            for j in range(self.cfg.DATA.NUM):
                frame_name, _, _, _, _ = self.df2[index-self.cfg.DATA.NUM//2+j]
                frame_one = np.load(frame_name)
                frame_one = frame_one[np.newaxis, :]
                if j == 0:
                    frame = frame_one
                else:
                    frame = np.concatenate((frame, frame_one), axis=0)
            frame = frame.transpose(3, 0, 1, 2)

        return torch.from_numpy(frame).float(), torch.tensor(label).long()

# TODO : STEP
class Crohn2016PrepareAvi:
    def __init__(self, cfg, batch, crop_size, train=True):
        self.batch = batch
        self.crop_size = crop_size
        self.train = train
        self.cfg = cfg
        self.step = cfg.DATA.STEP

    def __call__(self, avi_name, stomach_frame, s_intestine_frame, l_intestine_frame, i=None):
        # get frame
        tmp = cv2.VideoCapture(avi_name[0])
        # print(avi_name[0], tmp.isOpened())
        frame_num = tmp.get(7)
        frames = None
        labels = np.array([])
        if i is not None:
            if self.cfg.DATA.NUM == 1:
                i = int(i)
                i = min(i, frame_num-1)
                i = max(stomach_frame, i)
                tmp.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = tmp.read()
                if self.cfg.DATA.CROP == 128:
                    frame = frame[32:288, 32:288, :]
                    frame = skimage.measure.block_reduce(frame, (2, 2, 1), np.max)
                else:
                    # frame[:80, :100, :] = [0, 0, 0]
                    # frame[:30, 100:, :] = [0, 0, 0]
                    # frame[540:, :200, :] = [0, 0, 0]
                    
                    frame[305:, :131, :] = [0, 0, 0]
                    frame[:15, 260:, :] = [0, 0, 0]
                    
                frame = frame.transpose(2, 0, 1)
                frames = frame[np.newaxis, :]
            else:
                # index = random.randint(stomach_frame + self.cfg.DATA.NUM // 2, frame_num - (self.cfg.DATA.NUM*2))
                i = int(i)
                i = min(i, frame_num - 1 - self.cfg.DATA.NUM * self.step)
                i = max(stomach_frame + (self.cfg.DATA.NUM // 2 * self.step), i)
                index = i
                for j in range(self.cfg.DATA.NUM):
                    # tmp = cv2.VideoCapture(avi_name[0])
                    tmp.set(cv2.CAP_PROP_POS_FRAMES, int(index - (self.cfg.DATA.NUM // 2 * self.step) + (j * self.step)))
                    flag, frame_one = tmp.read()
                    # print('index', int(index - (self.cfg.DATA.NUM // 2 * self.step) + (j * self.step)))
                    # print(flag)
                    if self.cfg.DATA.CROP == 128:
                        frame_one = frame_one[32:288, 32:288, :]
                        frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
                    else:
                        # print('index:', j)
                        # print('ok')
                        # print(frame_one)
                        # print(frame_one.shape)
                        # frame_one[:80, :100, :] = [0, 0, 0]
                        # frame_one[:30, 100:, :] = [0, 0, 0]
                        # frame_one[540:, :200, :] = [0, 0, 0]
                        
                        frame_one[305:, :131, :] = [0, 0, 0]
                        frame_one[:15, 260:, :] = [0, 0, 0]
                        
                    frame_one = frame_one[np.newaxis, :]
                    # print(flag)
                    if j == 0:
                        frame = frame_one
                    else:
                        frame = np.concatenate((frame, frame_one), axis=0)
                # frames = frame.transpose(3, 0, 1, 2)[np.newaxis, :]
                frames = frame.transpose(0, 3, 1, 2)[np.newaxis, :]

            if i < s_intestine_frame:
                labels = np.append(labels, 0)
            elif i > l_intestine_frame:
                labels = np.append(labels, 2)
            else:
                labels = np.append(labels, 1)
        else:
            for batch_index in range(self.batch):
                # print(0, frame_num-1)
                if self.cfg.DATA.NUM == 1:
                    index = random.randint(stomach_frame, frame_num - 1)
                    tmp.set(cv2.CAP_PROP_POS_FRAMES, index)
                    flag, frame = tmp.read()
                    # print(flag)
                    if self.cfg.DATA.CROP == 128:
                        frame = frame[32:288, 32:288, :]
                        frame = skimage.measure.block_reduce(frame, (2, 2, 1), np.max)
                    else:
                        # frame[:80, :100, :] = [0, 0, 0]
                        # frame[:30, 100:, :] = [0, 0, 0]
                        # frame[540:, :200, :] = [0, 0, 0]
                        
                        frame[305:, :131, :] = [0, 0, 0]
                        frame[:15, 260:, :] = [0, 0, 0]
                        
                    frame = frame.transpose(2, 0, 1)[np.newaxis, :]
                else:
                    index = random.randint(stomach_frame + (self.cfg.DATA.NUM // 2 * self.step),
                                           frame_num - (self.cfg.DATA.NUM // 2 * self.step) - 2)
                    for j in range(self.cfg.DATA.NUM):
                        # tmp = cv2.VideoCapture(avi_name[0])
                        tmp.set(cv2.CAP_PROP_POS_FRAMES, int(index - (self.cfg.DATA.NUM // 2 * self.step) + (j * self.step)))
                        flag, frame_one = tmp.read()
                        if self.cfg.DATA.CROP == 128:
                            frame_one = frame_one[32:288, 32:288, :]
                            frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
                        else:
                            # frame_one[:80, :100, :] = [0, 0, 0]
                            # frame_one[:30, 100:, :] = [0, 0, 0]
                            # frame_one[540:, :200, :] = [0, 0, 0]
                            
                            frame_one[305:, :131, :] = [0, 0, 0]
                            frame_one[:15, 260:, :] = [0, 0, 0]
                            
                        frame_one = frame_one[np.newaxis, :]
                        # print(flag)
                        if j == 0:
                            frame = frame_one
                        else:
                            frame = np.concatenate((frame, frame_one), axis=0)
                    # print(frame.shape)
                    # frame = frame.transpose(0, 3, 1, 2)
                    # frame[305:, :131, :] = [0] * self.cfg.DATA.NUM * 3
                    # frame[:15, 260:, :] = [0] * self.cfg.DATA.NUM * 3
                    # frame = frame.transpose(3, 0, 1, 2)[np.newaxis, :]
                    frame = frame.transpose(0, 3, 1, 2)[np.newaxis, :]

                if batch_index == 0:
                    frames = frame
                else:
                    frames = np.concatenate((frames, frame), axis=0)
                # print(frame.shape, frames.shape)

                if index < s_intestine_frame:
                    labels = np.append(labels, 0)
                elif index > l_intestine_frame:
                    labels = np.append(labels, 2)
                else:
                    labels = np.append(labels, 1)

        # get label
        # 0: stomach, 1: small intestine, 2: large intestine
        # print(frames.shape, labels.shape)
        return torch.from_numpy(frames).float(), torch.from_numpy(labels).long()


class Crohn2016PrepareJpg:
    def __init__(self, cfg, batch, crop_size, train=True):
        self.batch = batch
        self.crop_size = crop_size
        self.train = train
        self.cfg = cfg

    def __call__(self, avi_name, stomach_frame, s_intestine_frame, l_intestine_frame, i=None):
        # get frame
        tmp = cv2.VideoCapture(avi_name[0])
        # print(avi_name[0], tmp.isOpened())
        stomach_frame = int(stomach_frame.item())
        frame_num = tmp.get(7)
        frames = None
        labels = np.array([])
        if i is not None:
            if self.cfg.DATA.NUM == 1:
                i = int(i)
                i = min(i, frame_num-1)
                i = max(stomach_frame, i)
                # tmp.set(cv2.CAP_PROP_POS_FRAMES, i)
                # _, frame = tmp.read()
                frame = Image.open(avi_name[0][:-4] + f"_{i}.jpg")
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if self.cfg.DATA.CROP == 128:
                    frame = frame[32:288, 32:288, :]
                    frame = skimage.measure.block_reduce(frame, (2, 2, 1), np.max)
                else:
                    frame[305:, :131, :] = [0, 0, 0]
                    frame[:15, 260:, :] = [0, 0, 0]
                frame = frame.transpose(2, 0, 1)
                frames = frame[np.newaxis, :]
            else:
                # index = random.randint(stomach_frame + self.cfg.DATA.NUM // 2, frame_num - self.cfg.DATA.NUM)
                i = int(i)
                i = min(i, frame_num - 1 - self.cfg.DATA.NUM)
                i = max(stomach_frame, i)
                index = i
                for j in range(self.cfg.DATA.NUM):
                    # tmp.set(cv2.CAP_PROP_POS_FRAMES, index - self.cfg.DATA.NUM // 2 + j)
                    # flag, frame_one = tmp.read()
                    frame = Image.open(avi_name[0][:-4] + f"_{index - self.cfg.DATA.NUM // 2 + j}.jpg")
                    frame = np.array(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if self.cfg.DATA.CROP == 128:
                        frame_one = frame_one[32:288, 32:288, :]
                        frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
                    else:
                        frame_one[305:, :131, :] = [0, 0, 0]
                        frame_one[:15, 260:, :] = [0, 0, 0]
                    frame_one = frame_one[np.newaxis, :]
                    # print(flag)
                    if j == 0:
                        frame = frame_one
                    else:
                        frame = np.concatenate((frame, frame_one), axis=0)
                frames = frame.transpose(3, 0, 1, 2)[np.newaxis, :]

            if i < s_intestine_frame:
                labels = np.append(labels, 0)
            elif i > l_intestine_frame:
                labels = np.append(labels, 2)
            else:
                labels = np.append(labels, 1)
        else:
            for batch_index in range(self.batch):
                # print(0, frame_num-1)
                if self.cfg.DATA.NUM == 1:
                    index = random.randint(stomach_frame, frame_num - 1)
                    # tmp.set(cv2.CAP_PROP_POS_FRAMES, index)
                    # flag, frame = tmp.read()
                    frame = Image.open(avi_name[0][:-4] + f"_{index}.jpg")
                    frame = np.array(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # print(flag)
                    if self.cfg.DATA.CROP == 128:
                        frame = frame[32:288, 32:288, :]
                        frame = skimage.measure.block_reduce(frame, (2, 2, 1), np.max)
                    else:
                        frame[305:, :131, :] = [0, 0, 0]
                        frame[:15, 260:, :] = [0, 0, 0]
                    frame = frame.transpose(2, 0, 1)[np.newaxis, :]
                else:
                    index = random.randint(stomach_frame + self.cfg.DATA.NUM//2, frame_num - self.cfg.DATA.NUM)
                    for j in range(self.cfg.DATA.NUM):
                        # tmp.set(cv2.CAP_PROP_POS_FRAMES, index - self.cfg.DATA.NUM//2 + j)
                        # flag, frame_one = tmp.read()
                        frame_one = Image.open(avi_name[0][:-4] + f"_{index - self.cfg.DATA.NUM//2 + j}.jpg")
                        frame_one = np.array(frame_one)
                        frame_one = cv2.cvtColor(frame_one, cv2.COLOR_RGB2BGR)
                        if self.cfg.DATA.CROP == 128:
                            frame_one = frame_one[32:288, 32:288, :]
                            frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
                        frame_one = frame_one[np.newaxis, :]
                        # print(flag)
                        if j == 0:
                            frame = frame_one
                        else:
                            frame = np.concatenate((frame, frame_one), axis=0)
                    # print(frame.shape)
                    # frame = frame.transpose(0, 3, 1, 2)
                    # frame[305:, :131, :] = [0] * self.cfg.DATA.NUM * 3
                    # frame[:15, 260:, :] = [0] * self.cfg.DATA.NUM * 3
                    frame = frame.transpose(3, 0, 1, 2)[np.newaxis, :]

                if batch_index == 0:
                    frames = frame
                else:
                    frames = np.concatenate((frames, frame), axis=0)
                # print(frame.shape, frames.shape)

                if index < s_intestine_frame:
                    labels = np.append(labels, 0)
                elif index > l_intestine_frame:
                    labels = np.append(labels, 2)
                else:
                    labels = np.append(labels, 1)

        # get label
        # 0: stomach, 1: small intestine, 2: large intestine
        # print(frames.dtype, labels.dtype)
        return torch.from_numpy(frames).float(), torch.from_numpy(labels).long()


class Crohn2016Prepare:
    def __init__(self, cfg, batch, crop_size, train=True):
        self.batch = batch
        self.crop_size = crop_size
        self.train = train
        self.cfg = cfg

    def __call__(self, avi_name, stomach_frame, s_intestine_frame, l_intestine_frame, i=None):
        # stomach_time = stomach_time[0]
        # s_intestine_time = s_intestine_time[0]
        # l_intestine_time = l_intestine_time[0]
        # print(avi_name, stomach_frame, s_intestine_frame, l_intestine_frame)
        # print('=========>', stomach_time[-2:])
        # stomach_frame = 3 * int(stomach_time[-2:]) + 180 * int(stomach_time[3:5]) + 10800 * int(stomach_time[0:2])
        # s_f = 3 * int(s_intestine_time[-2:]) + 180 * int(s_intestine_time[3:5]) + 10800 * int(s_intestine_time[0:2])
        # l_f = 3 * int(l_intestine_time[-2:]) + 180 * int(l_intestine_time[3:5]) + 10800 * int(l_intestine_time[0:2])
        # print(stomach_frame, s_f, l_f)

        # get frame
        tmp = cv2.VideoCapture(avi_name[0])
        # print(tmp.isOpened())
        frame_num = tmp.get(7)
        # print(tmp.get(0), tmp.get(1), tmp.get(2), tmp.get(3), tmp.get(4), tmp.get(5), tmp.get(6), tmp.get(7))
        # fps = tmp.get(5)
        # frames = np.array([])
        frames = None
        labels = np.array([])
        if i is not None:
            i = int(i)
            i = min(i, frame_num-1)
            i = max(stomach_frame, i)
            frame = np.load(f'{avi_name[0][:-4]}_{i}.npy')
            frames = frame[np.newaxis, :]

            if i < s_intestine_frame:
                labels = np.append(labels, 0)
            elif i > l_intestine_frame:
                labels = np.append(labels, 2)
            else:
                labels = np.append(labels, 1)
        else:
            for batch_index in range(self.batch):
                # print(0, frame_num-1)

            # def do_batch(batch_index):

                if self.cfg.DATA.NUM == 1:
                    index = random.randint(stomach_frame, frame_num - 1)
                    # tmp.set(cv2.CAP_PROP_POS_FRAMES, index)
                    # flag, frame = tmp.read()
                    # print(flag)
                    # frame[305:, :131, :] = [0, 0, 0]
                    # frame[:15, 260:, :] = [0, 0, 0]
                    frame = np.load(f'{avi_name[0][:-4]}_{index}.npy')
                    frame = frame.transpose(2, 0, 1)[np.newaxis, :]
                else:
                    index = random.randint(stomach_frame + self.cfg.DATA.NUM//2, frame_num - self.cfg.DATA.NUM)
                    for j in range(self.cfg.DATA.NUM):
                        # tmp.set(cv2.CAP_PROP_POS_FRAMES, index - self.cfg.DATA.NUM//2 + j)
                        # flag, frame_one = tmp.read()
                        # if self.cfg.DATA.CROP == 128:
                        #     frame_one = frame_one[32:288, 32:288, :]
                        #     frame_one = skimage.measure.block_reduce(frame_one, (2, 2, 1), np.max)
                        frame_one = np.load(f'{avi_name[0][:-4]}_{index - self.cfg.DATA.NUM//2 + j}.npy')
                        frame_one = frame_one[np.newaxis, :]
                        # print(flag)
                        if j == 0:
                            frame = frame_one
                        else:
                            frame = np.concatenate((frame, frame_one), axis=0)
                    # print(frame.shape)
                    # frame = frame.transpose(0, 3, 1, 2)
                    # frame[305:, :131, :] = [0] * self.cfg.DATA.NUM * 3
                    # frame[:15, 260:, :] = [0] * self.cfg.DATA.NUM * 3
                    frame = frame.transpose(3, 0, 1, 2)[np.newaxis, :]

                if batch_index == 0:
                    frames = frame
                else:
                    frames = np.concatenate((frames, frame), axis=0)
                #
                if index < s_intestine_frame:
                #    label = 0
                     labels = np.append(labels, 0)
                elif index > l_intestine_frame:
                #    label = 2
                     labels = np.append(labels, 2)
                else:
                #    label = 1
                     labels = np.append(labels, 1)
                # return frame, label

            # param = range(self.batch)
            # p = multiprocessing.Pool(self.cfg.SYSTEM.NUM_WORKERS)
            # b = p.map(do_batch, param)
            # p.close()
            # p.join()
            # print(b)

        # get label
        # 0: stomach, 1: small intestine, 2: large intestine
        # print(frames.dtype, labels.dtype)
        return torch.from_numpy(frames).float(), torch.from_numpy(labels).long()

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            if self.cfg.DATA.RESIZE is False:
                x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            if self.cfg.DATA.RESIZE is False:
                x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        # print(height, width, depth, crop_size)
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())


def split_dataset(data_root, nfold=4, seed=42, select=0):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "BraTS19*"))
    n_patients = len(patients_dir)
    print(f"total patients: {n_patients}")
    pid_idx = np.arange(n_patients)
    np.random.seed(seed)
    np.random.shuffle(pid_idx)
    n_fold_list = np.array_split(pid_idx, nfold)
    print(f"split {len(n_fold_list)} folds and every fold have {len(n_fold_list[0])} patients")
    val_patients_list = []
    train_patients_list = []
    for i, fold in enumerate(n_fold_list):
        if i == select:
            for idx in fold:
                val_patients_list.append(patients_dir[idx])
        else:
            for idx in fold:
                train_patients_list.append(patients_dir[idx])
    print(f"train patients: {len(train_patients_list)}, test patients: {len(val_patients_list)}")

    return train_patients_list, val_patients_list


def read_fold(subset_dir):
    lst = []
    f = os.listdir(os.path.join(subset_dir, 'image'))
    for name in f:
        cts = os.listdir(os.path.join(subset_dir, 'image', name))
        for ct in cts:
            lst.append({'name': name,
                        'ct': ct})
    # f = os.listdir(os.path.join(subset_dir))
    # for name in f:
    #     cts = os.listdir(os.path.join(subset_dir, name, 'image'))
    #     for ct in cts:
    #         lst.append({'name': name,
    #                     'ct': ct})
    random.seed(42)
    random.shuffle(lst)
    return lst


def read_fold_sts(subset_dir, subset, cfg):
    lst_name = []
    f = sorted(os.listdir(os.path.join(subset_dir)))
    random.seed(42)
    random.shuffle(f)
    num = [0, 13, 26, 39, 52]
    train_list = list(set(range(52)) - set(range(num[cfg.TEST.FOLD], num[cfg.TEST.FOLD+1])))

    for name in f:
        lst = []
        # print(f, name)
        if name[-2:] == 'xt':
            continue
        cur_name = int(name[-2:])
        if subset is 'test':
            if cur_name in range(num[cfg.TEST.FOLD], num[cfg.TEST.FOLD+1]):
                cts = os.listdir(os.path.join(subset_dir, name, 'masks'))
                for ct in cts:
                    if ct[-3:] == 'npy':
                        mask = np.load(os.path.join(subset_dir, name, 'masks', ct))
                        if np.max(mask) > 0:
                            lst.append({'name': name, 'ct': ct[5:]})
                lst_name.append(lst)
        elif subset is 'valid':  # 'train' or 'val'
            if cur_name in train_list[-3:]:
                cts = os.listdir(os.path.join(subset_dir, name, 'masks'))
                for ct in cts:
                    if ct[-3:] == 'npy':
                        mask = np.load(os.path.join(subset_dir, name, 'masks', ct))
                        if np.max(mask) > 0:
                            lst.append({'name': name, 'ct': ct[5:]})
                lst_name.append(lst)
        else:
            if cur_name in train_list[:-3]:
                cts = os.listdir(os.path.join(subset_dir, name, 'masks'))
                for ct in cts:
                    if ct[-3:] == 'npy':
                        mask = np.load(os.path.join(subset_dir, name, 'masks', ct))
                        if np.max(mask) > 0:
                            lst.append({'name': name, 'ct': ct[5:]})
                lst_name.append(lst)

    return lst_name


def get_test(cfg):
    csv = cfg.DIRS.DATA_CSV
    dts = Crohn2016(cfg, csv, mode="test")
    batch_size = cfg.VAL.BATCH_SIZE
    dataloader = DataLoader(dts, batch_size=1,
                            shuffle=False, drop_last=False,
                            num_workers=cfg.SYSTEM.NUM_WORKERS)
    data_prepare = Crohn2016PrepareAvi(cfg, batch_size, crop_size=cfg.VAL.CROP, train=False)

    return dataloader, data_prepare


def get_dataset(Mode, cfg):
    if Mode == 'train':
        csv = cfg.DIRS.DATA_CSV
        if cfg.DATA.NAME == "Crohn2016":
            dts = Crohn2016(cfg, csv, mode="train")
        elif cfg.DATA.NAME == "Crohn2016Npy":
            dts = Crohn2016Npy(cfg, csv, train=True)
        elif cfg.DATA.NAME == "Crohn2016_jpg":
            dts = Crohn2016_jpg(cfg, csv, mode="train")
        elif cfg.DATA.NAME == "Crohn15to23":
            dts = Crohn15to23(cfg, csv, mode="train")
        else:
            print('error')
        batch_size = cfg.TRAIN.BATCH_SIZE
        data_prepare = None
        if cfg.DATA.NPY is True:
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=True, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016Prepare(cfg, batch_size, crop_size=cfg.TRAIN.CROP, train=True)
        elif cfg.DATA.JPG is True:
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=True, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016PrepareJpg(cfg, batch_size, crop_size=cfg.TRAIN.CROP, train=True)
        elif cfg.DATA.NAME == "Crohn2016_jpg":
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=True, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
        elif cfg.DATA.NAME == "Crohn15to23":
            dataloader = DataLoaderX(dts, batch_size=batch_size,
                                    shuffle=True, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
        else:
            dataloader = DataLoader(dts, batch_size=1,
                                    shuffle=True, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016PrepareAvi(cfg, batch_size, crop_size=cfg.TRAIN.CROP, train=True)

    elif Mode == "valid":
        csv = cfg.DIRS.DATA_CSV
        if cfg.DATA.NAME == "Crohn2016":
            dts = Crohn2016(cfg, csv, mode="val")
        elif cfg.DATA.NAME == "Crohn2016Npy":
            dts = Crohn2016Npy(cfg, csv, train=False)
        elif cfg.DATA.NAME == "Crohn2016_jpg":
            dts = Crohn2016_jpg(cfg, csv, mode="val")
        elif cfg.DATA.NAME == "Crohn15to23":
            dts = Crohn15to23(cfg, csv, mode="val")
        else:
            print('error')
        batch_size = cfg.VAL.BATCH_SIZE
        data_prepare = None
        if cfg.DATA.NPY is True:
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016Prepare(cfg, batch_size, crop_size=cfg.VAL.CROP, train=False)
        elif cfg.DATA.JPG is True:
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016PrepareJpg(cfg, batch_size, crop_size=cfg.VAL.CROP, train=False)
        elif cfg.DATA.NAME == "Crohn2016_jpg":
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
        elif cfg.DATA.NAME == "Crohn15to23":
            dataloader = DataLoaderX(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
        else:
            dataloader = DataLoader(dts, batch_size=1,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016PrepareAvi(cfg, batch_size, crop_size=cfg.VAL.CROP, train=False)
    elif Mode == "test":
        csv = cfg.DIRS.DATA_CSV
        # if cfg.DATA.NAME == "Crohn2016":
        if cfg.DATA.NAME == "Crohn2016":
            dts = Crohn2016(cfg, csv, mode="test")
        elif cfg.DATA.NAME == "Crohn2016_jpg":
            dts = Crohn2016_jpg(cfg, csv, mode="test")
        elif cfg.DATA.NAME == "Crohn15to23":
            dts = Crohn15to23(cfg, csv, mode="test")
        # elif cfg.DATA.NAME == "Crohn2016Npy":
        #     dts = Crohn2016Npy(cfg, csv, train=False)
        # else:
        #  print('error')
        batch_size = cfg.TEST.BATCH_SIZE
        data_prepare = None
        if cfg.DATA.NPY is True:
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016Prepare(cfg, batch_size, crop_size=cfg.VAL.CROP, train=False)
        elif cfg.DATA.NPY is True:
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016PrepareJpg(cfg, batch_size, crop_size=cfg.VAL.CROP, train=False)
        elif cfg.DATA.NAME == "Crohn2016_jpg":
            dataloader = DataLoader(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
        elif cfg.DATA.NAME == "Crohn15to23":
            dataloader = DataLoaderX(dts, batch_size=batch_size,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
        else:
            dataloader = DataLoader(dts, batch_size=1,
                                    shuffle=False, drop_last=False,
                                    num_workers=cfg.SYSTEM.NUM_WORKERS)
            data_prepare = Crohn2016PrepareAvi(cfg, batch_size, crop_size=cfg.VAL.CROP, train=False)

    else:
        raise Exception("ERROR in get_dataset()")
    return dataloader, data_prepare


def get_debug_dataset(mode, cfg):
    # cfg = get_cfg_defaults()
    if mode == 'train':
        csv = os.path.join(cfg.TRAIN.CSV, f"train_fold{cfg.TRAIN.FOLD}.csv")
        dts = Crohn2016(cfg, csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 5))
        # dts = Subset(dts)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size,
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    else:
        csv = os.path.join(cfg.VAL.CSV, f"val_fold{cfg.VAL.FOLD}.csv")
        dts = Crohn2016(cfg, csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 2))
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size,
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader


if __name__ == "__main__":
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
    print(dataiter1.next(), dataiter1.next(), dataiter1.next(), dataiter1.next())
    print(dataiter1.next())

    avi_name, _, _, _ = dataiter1.next()
    print(avi_name)
    tmp = cv2.VideoCapture(avi_name[0])
    tmp.set(cv2.CAP_PROP_POS_FRAMES, 500)
    a, b = tmp.read()
    print(b.shape, b[160, 160], tmp.get(5))
    print(type(b))
    bb = np.array([])
    np.append(bb, b)
    bb = torch.from_numpy(bb)
    bb.cuda()