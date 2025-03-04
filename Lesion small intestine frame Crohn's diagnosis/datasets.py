import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import pandas as pd
import glob
from PIL import Image
import skimage.measure
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

################################################### MG ##############################################
import random
import copy

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, prob=0.5):
    # augmentation by flipping
    cnt = 2
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        cnt = cnt - 1

    return x


def data_augmentation2(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 32
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        block_noise_size_z = random.randint(1, img_deps // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        noise_z = random.randint(0, img_deps - block_noise_size_z)
        window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y,
                 noise_z:noise_z + block_noise_size_z,
                 ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 2
    while cnt > 0 and random.random() < 0.65:
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
        block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                               block_noise_size_y,
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x


def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
    x[:,
    noise_x:noise_x + block_noise_size_x,
    noise_y:noise_y + block_noise_size_y,
    noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                            noise_y:noise_y + block_noise_size_y,
                                            noise_z:noise_z + block_noise_size_z]
    cnt = 2
    while cnt > 0 and random.random() < 0.65:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt -= 1
    return x


################################################### MG ##############################################

# Crohn_all
class CrohnFrameAll(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_frame_all init', mode)
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        self.csv = cfg.DIRS.DATA_CSV
        try:
            self.df = pd.read_csv(self.csv)
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv, encoding="gbk")

        mode_map = {'train': 'Train', 'val': 'Valid', 'test': 'Test'}
        # mode_map = {'train': 'Val', 'val': 'Val', 'test': 'Val'}
        mode = mode_map[self.mode]
        num_videos = 0
        stat = {'Crohn': 0, 'Health': 0, 'Crohn_p': 0, 'Health_p': 0}
        # sall = []
        # for flag in ['Crohn', 'Health']:
        #     for i in os.listdir(os.path.join(self.cfg.DIRS.DATA, flag)):
        #         sall.append(i)
        # sall = set(sall)

        for i in range(len(self.df)):
            datum = self.df.iloc[i]
            if datum['Fold'] == mode:
                flag = 'Crohn' if datum['Crohn'] == 1 else 'Health'
                name = datum['Name']
                if name not in os.listdir(self.cfg.DIRS.DATA):
                    continue
                pics = os.listdir(os.path.join(self.cfg.DIRS.DATA, name))
                # if mode == 'Train':
                pics = random.choices(pics, k=min(50, len(pics)))
                print(name, flag, len(pics))
                for pic in pics:
                    if ".jpg" in pic:
                        self.data.append([os.path.join(self.cfg.DIRS.DATA, name, pic), flag])
                        stat[flag] += 1
                stat[str(flag) + '_p'] += 1

                # for flag in ['Crohn', 'Health']:
                #     name = datum['CN_Name'] 
                #     if name not in os.listdir(os.path.join(self.cfg.DIRS.DATA, flag)):
                #         continue
                #     pics = os.listdir(os.path.join(self.cfg.DIRS.DATA, flag, name))
                #     # if mode == 'Train':
                #     #     pics = random.choices(pics, k=min(50, len(pics)))
                #     # print(name, flag, len(pics))
                #     for pic in pics:
                #         if ".jpg" in pic:
                #             self.data.append(os.path.join(self.cfg.DIRS.DATA, flag, name, pic))
                #             stat[flag] += 1
                #     stat[str(flag) + '_p'] += 1
                num_videos += 1
                # if name in sall:
                #     sall.remove(name)

        self.transform = A.Compose(
            [A.CenterCrop(cfg.DATA.SIZE, cfg.DATA.SIZE), A.Normalize(), ToTensorV2()])
        if self.cfg.TRAIN.AUGMENTATION and self.mode == 'train':
            # print('use TRAIN.AUGMENTATION')
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.ElasticTransform(p=0.5),
                A.CoarseDropout(p=0.5),
                A.RandomCrop(cfg.DATA.SIZE, cfg.DATA.SIZE),
                A.Normalize(),
                ToTensorV2()
            ])

        print(len(self.data), num_videos, stat)
        # t = 0
        # for name in sall:
        #     t += 1
        #     flag = 'Train'
        #     if t < 36:
        #         flag = 'Test'
        #     elif t < 72:
        #         flag = 'Val'
        #     print(f'{t},{name.split(" ")[-1]},,{name},,,,,,,NaN,{flag}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        frame_name = self.data[index][0]
        image = cv2.imread(frame_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] == 360:
            image = image[20:-20, 20:-20, :]
        # image[305:, :131, :] = [0, 0, 0]
        # image[:15, 260:, :] = [0, 0, 0]

        label_name = self.data[index][1]
        if "Crohn" == label_name:
            label = 1
        elif "Health" == label_name:
            label = 0
        else:
            raise KeyError('LABEL ERROR', frame_name, label_name)

        # tranform

        # frame = image.transpose(2, 0, 1)
        # if self.mode == "train":
        #     img_data = frame.astype(np.float32)
        #     img_data = data_augmentation(img_data, 0.2)
        #     img_data = np.expand_dims(img_data, 0)
        #     img_data = local_pixel_shuffling(img_data, prob=0.0)
        #     img_data = nonlinear_transformation(img_data, 0.2)
        #     # Inpainting & Outpainting
        #     if random.random() < 0.0:  # config.paint_rate
        #         if random.random() < 0.2:  # config.inpaint_rate
        #             # Inpainting
        #             img_data = image_in_painting(img_data)
        #         else:
        #             # Outpainting
        #             img_data = image_out_painting(img_data)
        #
        #     frame = img_data.astype(np.float32)
        #     frame = np.squeeze(frame, 0)
        #     frame = torch.from_numpy(frame).float()

        frame = self.transform(image=image)
        frame = frame['image']

        return frame, torch.tensor(label).long(), frame_name


class CrohnFrame(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_frame init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        self.transform = A.Compose(
            [A.CenterCrop(cfg.DATA.SIZE, cfg.DATA.SIZE), A.Normalize(), ToTensorV2()])
        if self.mode == "train":
            print("TRAIN SAMPLES")
            for flag in ['True', 'False']:
                for name in os.listdir(os.path.join(self.cfg.DIRS.DATA, 'Train', flag)):
                    for pic in os.listdir(os.path.join(self.cfg.DIRS.DATA, 'Train', flag, name)):
                        if ".jpg" in pic:
                            self.data.append(os.path.join(self.cfg.DIRS.DATA, 'Train', flag, name, pic))
        else:
            print("TEST SAMPLES")
            for flag in ['True', 'False']:
                for name in os.listdir(os.path.join(self.cfg.DIRS.DATA, 'Valid', flag)):
                    for pic in os.listdir(os.path.join(self.cfg.DIRS.DATA, 'Valid', flag, name)):
                        if ".jpg" in pic:
                            self.data.append(os.path.join(self.cfg.DIRS.DATA, 'Valid', flag, name, pic))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        frame_name = self.data[index]
        image = cv2.imread(frame_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            image = image[20:-20, 20:-20, :]
        except TypeError:
            print(frame_name)
            image = np.zeros((320, 320, 3))
        # frame_one = frame_one.transpose(2, 0, 1)

        # tranform
        frame = self.transform(image=image)
        frame = frame['image']

        if "True" in frame_name:
            label = 1
        elif "False" in frame_name:
            label = 0
        else:
            raise KeyError('LABEL ERROR')

        return frame, torch.tensor(label).long(), frame_name


class CrohnSeqFrame(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_frame init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        self.seqs = os.listdir(cfg.DIRS.DATA_SEQ)
        self.dropout = cfg.TRAIN.DROPOUT
        self.transform = A.Compose(
            [A.CenterCrop(cfg.DATA.SIZE, cfg.DATA.SIZE), A.Normalize(), ToTensorV2()])
        if self.cfg.TRAIN.AUGMENTATION and self.mode == 'train':
            # print('use TRAIN.AUGMENTATION')
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.ElasticTransform(p=0.5),
                A.CoarseDropout(p=0.5),
                A.RandomCrop(cfg.DATA.SIZE, cfg.DATA.SIZE),
                A.Normalize(),
                ToTensorV2()
            ])

        self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
        for i in range(len(self.df)):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                sample_name = datum['Name']
                label = datum['Crohn']
                if str(sample_name + '.npy') in self.seqs:
                    seq_2000 = list(np.load(os.path.join(cfg.DIRS.DATA_SEQ, str(sample_name + '.npy'))))
                    self.data.append((seq_2000, label))
        print('Length of Data:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (seq_2000, label) = self.data[index]
        name = seq_2000[0][1].split('/')[-2]
        frames = None

        if self.dropout:
            if self.mode == 'train':
                seqs = random.sample(seq_2000, len(seq_2000) * self.dropout)
            else:
                seqs = seq_2000
                del seqs[::int(1 / self.dropout)]
        else:
            seqs = seq_2000

        for i, j in enumerate(seqs):
            frame_name = j[1]
            image = cv2.imread(frame_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] == 360:
                image = image[20:-20, 20:-20, :]

            frame = self.transform(image=image)
            frame = frame['image']

            frame_one = frame[np.newaxis, :]
            if frames is None:
                frames = frame_one
            else:
                frames = np.concatenate((frames, frame_one), axis=0)

        return frames, torch.tensor(label).long(), name


class CrohnSeq(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_frame init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        self.seqs = os.listdir(cfg.DIRS.DATA_SEQ)
        self.dropout = cfg.TRAIN.DROPOUT
        self.transform = A.Compose(
            [A.CenterCrop(cfg.DATA.SIZE, cfg.DATA.SIZE), A.Normalize(), ToTensorV2()])

        self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
        for i in range(len(self.df)):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                sample_name = datum['Name']
                # print(sample_name)
                label = datum['Crohn']
                if str(sample_name) + '.npy' in self.seqs:
                    seq_2000 = list(np.load(os.path.join(cfg.DIRS.DATA_SEQ, str(sample_name + '.npy'))))
                    for k in seq_2000:
                        self.data.append((k[1], label))
        print('Length of Data:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        frame_name, label = self.data[index]
        name = frame_name.split('/')[-2]

        image = cv2.imread(frame_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] == 360:
            image = image[20:-20, 20:-20, :]

        frame = self.transform(image=image)
        frame = frame['image']

        # label = -1

        return frame, torch.tensor(label).long(), name


class CrohnSeqPred(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_seq_pred init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        # self.seqs = os.listdir(cfg.DIRS.DATA_SEQ)
        self.dropout = cfg.TRAIN.DROPOUT
        try:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV, encoding="gbk")

        for i in range(len(self.df) - 1, -1, -1):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                name = datum['Name']
                if str(name) + '_0.npy' not in os.listdir(self.cfg.DIRS.DATA):
                    self.df.drop(i, inplace=True, axis=0)
            else:
                self.df.drop(i, inplace=True, axis=0)

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
            print("TRAIN SAMPLES")
            print(self.df)
        else:
            self.flod = cfg.TEST.FOLD
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
            print("VAL/TEST SAMPLES")
            print(self.df)

        for i in range(len(self.df)):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                name = datum['Name']
                if str(name) + '_0.npy' in os.listdir(self.cfg.DIRS.DATA):
                    self.data.append((os.path.join(self.cfg.DIRS.DATA, name + '_0.npy'), datum['Crohn']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (seqs, label) = self.data[index]
        frames = np.load(seqs)
        # idx = list(range(0, 2000, 2))
        if self.mode == 'train':
            rd = np.random.randint(0, 4)
            idx = list(range(0 + rd, 2000 + rd - 4, 4))
            # p = 1
            # len_p = 500 // p
            # for i in range(p):
            #     if np.random.randint(0, 10) > 9:
            #         idx[i * len_p:(i + 1) * len_p] = np.random.randint(i * len_p * 4, (i + 1) * len_p * 4, size=len_p)
        elif self.mode == 'val':
            idx = list(range(0, 2000, 4))
        else:
            idx = list(range(0, 2000, 4))
            frames1 = frames[idx]
            idx = list(range(1, 2000, 4))
            frames2 = frames[idx]
            idx = list(range(2, 2000, 4))
            frames3 = frames[idx]
            idx = list(range(3, 2000, 4))
            frames4 = frames[idx]
            name = seqs.split('/')[-1][:-4]
            return (torch.from_numpy(frames1).float(),
                    torch.from_numpy(frames2).float(),
                    torch.from_numpy(frames3).float(),
                    torch.from_numpy(frames4).float(),
                    ), torch.tensor(label).long(), name
        frames = frames[idx]
        name = seqs.split('/')[-1][:-4]

        return torch.from_numpy(frames).float(), torch.tensor(label).long(), name


class CrohnSeqFeatPre(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_seq_feature_prepare init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        self.dropout = cfg.TRAIN.DROPOUT
        try:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV, encoding="gbk")

        for i in range(len(self.df) - 1, -1, -1):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                name = datum['Name']
                if str(name) + '.npy' not in os.listdir(self.cfg.DIRS.DATA):
                    self.df.drop(i, inplace=True, axis=0)
            else:
                self.df.drop(i, inplace=True, axis=0)

        for i in range(len(self.df)):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                name = datum['Name']
                # if name == 'u8f69 2019_05_07':
                #     continue
                if str(name) + '.npy' in os.listdir(self.cfg.DIRS.DATA):
                    num = cfg.TEST.BATCH_SIZE
                    patch = cfg.TEST.PRED
                    print('loading', name)
                    frame_list_all = np.load(os.path.join(self.cfg.DIRS.DATA, name + '.npy'))
                    # assert len(frame_list_all) > 2500, name
                    if patch > 0:
                        for FOLD in range(patch):
                            idx = FOLD
                            # print(frame_list)
                            len_part = len(frame_list_all) // patch # 分成四段
                            frame_list = frame_list_all[idx * len_part: (idx + 1) * len_part]
                            # frame_list = sorted(frame_list)
                            frame_scores = frame_list[:, 0].astype(np.float)
                            frame_list = frame_list[frame_scores.argsort()[::-1][:num]]
                            ######################
                            frame_list_idx = [ii[1].split('/')[-1][:-4] for ii in frame_list]
                            # print(frame_list, len(frame_list))
                            os.makedirs(cfg.DIRS.OUTPUTS, exist_ok=True)
                            os.makedirs(os.path.join(cfg.DIRS.OUTPUTS, name), exist_ok=True)
                            np.save(os.path.join(cfg.DIRS.OUTPUTS, name,
                                                 f'idx_top_{cfg.TEST.BATCH_SIZE}_idx_{FOLD}_of_{cfg.TEST.PRED}.npy'),
                                    frame_list_idx)
                            # a = 0 / 0
                            for j in frame_list:
                                self.data.append((j[1], idx))
                    else:
                        num = min(len(frame_list_all), 2000)
                        frame_list = frame_list_all
                        frame_scores = frame_list[:, 0].astype(np.float)
                        frame_list = frame_list[frame_scores.argsort()[::-1][:num]]
                        ######################
                        frame_list_idx = [ii[1].split('/')[-1][:-4] for ii in frame_list]
                        # print(frame_list, len(frame_list))
                        os.makedirs(cfg.DIRS.OUTPUTS, exist_ok=True)
                        os.makedirs(os.path.join(cfg.DIRS.OUTPUTS, name), exist_ok=True)
                        np.save(os.path.join(cfg.DIRS.OUTPUTS, name, f'idx.npy'), frame_list_idx)
                        for j in frame_list:
                            self.data.append((j[1], -1))

        self.transform = A.Compose(
            # [A.CenterCrop(cfg.DATA.SIZE, cfg.DATA.SIZE), A.Normalize(), ToTensorV2()])
            [A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        frame_name, idx = self.data[index]
        name = frame_name.split('/')[-2]

        image = cv2.imread(frame_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] == 360:
            image = image[20:-20, 20:-20, :]
        else:
            image = image

        # tranform
        frame = self.transform(image=image)
        frame = frame['image']

        label = idx

        return frame, torch.tensor(label).long(), name


class CrohnSeqFeat_backup(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_seq_feature init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        self.dropout = cfg.TRAIN.DROPOUT
        try:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV, encoding="gbk")

        for i in range(len(self.df) - 1, -1, -1):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                name = datum['Name']
                if str(name) not in os.listdir(self.cfg.DIRS.DATA):
                    self.df.drop(i, inplace=True, axis=0)
            else:
                self.df.drop(i, inplace=True, axis=0)

        if self.mode in ["train"]:
            self.flod = cfg.TRAIN.FOLD
            len_fold = len(self.df) // 5
            if self.flod == 0:
                self.df = self.df[len_fold:]
            elif self.flod == 1:
                self.df = pd.concat([self.df[:len_fold * 1], self.df[len_fold * 2:]])
            elif self.flod == 2:
                self.df = pd.concat([self.df[:len_fold * 2], self.df[len_fold * 3:]])
            elif self.flod == 3:
                self.df = pd.concat([self.df[:len_fold * 3], self.df[len_fold * 4:]])
            elif self.flod == 4:
                self.df = self.df[:len_fold * 4]
            else:
                raise Exception("Flod ERROR")
            print("TRAIN SAMPLES")
            print(self.df)
        else:
            self.flod = cfg.TEST.FOLD
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
            print("VAL/TEST SAMPLES")
            print(self.df)

        for i in range(len(self.df)):
            datum = self.df.iloc[i]
            if datum['Crohn'] in [0, 1]:
                name = datum['Name']
                if str(name) in os.listdir(self.cfg.DIRS.DATA):
                    s_frame = datum['S_intestine_frame']
                    l_frame = datum['L_intestine_frame']
                    self.data.append((name, datum['Crohn'], s_frame, l_frame))
        print('len:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name, label, s_frame, l_frame = self.data[index]

        mode = '-1'  # '64'
        pnum = 1  # 32

        frames = []
        idxes = []
        for i in range(pnum):
            if mode == '200':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_200_idx_{i}_of_4.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_200_idx_{i}_of_4.npy')).astype(np.float)
            elif mode == '64':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_64_idx_{i}_of_32.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_64_idx_{i}_of_32.npy')).astype(np.float)
            elif mode == '500':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_500_idx_{i}_of_4.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_500_idx_{i}_of_4.npy')).astype(np.float)
                # assert len(frame) == len(idx), f"{name}, {i}, {len(frame)}, {len(idx)}, {label}"  # TODO
                if self.mode == 'train':
                    len_t = min(len(frame), len(idx))
                    t = np.random.randint(0, len_t, size=250)
                    frame = frame[t]
                    idx = idx[t]
                else:
                    frame = frame[:250]
                    idx = idx[:250]
            elif mode == '-1':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'features.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx.npy')).astype(np.float)

                if self.mode == 'train':
                    len_t = min(len(frame), 500)
                    t = np.random.randint(0, len(frame), size=len_t)
                    frame = frame[t]
                    idx = idx[t]
                else:
                    len_t = min(len(frame), 500)
                    t = np.random.randint(0, len(frame), size=len_t)
                    frame = frame[t]
                    idx = idx[t]
                    # frame = frame[:len_t]
                    # idx = idx[:len_t]
            else:
                raise KeyError
            idx = (idx - s_frame) / (l_frame - s_frame)
            frames.append(frame)
            idxes.append(idx)
        frames = np.array(frames)
        idxes = np.array(idxes)

        return torch.from_numpy(frames).float(), torch.from_numpy(idxes).float(), torch.tensor(label).long(), name


class CrohnSeqFeat(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'Crohn_seq_feature init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        self.dropout = cfg.TRAIN.DROPOUT

        '''
        遇到字符编码错误,则是使用gbk编码重新读取
        '''
        try:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV, encoding="gbk")

        mode_map = {'train': 'Train', 'val': 'Valid', 'test': 'Test'}
        mode = mode_map[self.mode]

        # 添加当前mode的数据到self.data列表中
        for i in range(len(self.df)):
            datum = self.df.iloc[i]
            if datum['Fold'] == mode:
                name = datum['Name']
                if str(name) in os.listdir(self.cfg.DIRS.DATA):
                    s_frame = datum['S_intestine_frame']
                    l_frame = datum['L_intestine_frame']
                    self.data.append((name, datum['Crohn'], s_frame, l_frame))
        print('len:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name, label, s_frame, l_frame = self.data[index]

        mode = '500'  # '64'
        pnum = 4  # 32

        # 用来存储每个阶段的帧数据和对应的索引
        frames = []
        idxes = []
        for i in range(pnum):
            if mode == '200':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_200_idx_{i}_of_4.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_200_idx_{i}_of_4.npy')).astype(np.float)
            elif mode == '64':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_64_idx_{i}_of_32.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_64_idx_{i}_of_32.npy')).astype(np.float)
            elif mode == '500':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_500_idx_{i}_of_4.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_500_idx_{i}_of_4.npy')).astype(np.float)
                # assert len(frame) == len(idx), f"{name}, {i}, {len(frame)}, {len(idx)}, {label}"  # TODO
                if self.mode == 'train':
                    len_t = min(len(frame), len(idx))
                    t = np.random.randint(0, len_t, size=350)
                    frame = frame[t]
                    idx = idx[t]
                else:
                    frame = frame[:350]
                    idx = idx[:350]
            elif mode == '-1':
                frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'features.npy'))
                idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx.npy')).astype(np.float)

                if self.mode == 'train':
                    len_t = min(len(frame), 500)
                    t = np.random.randint(0, len(frame), size=len_t)
                    frame = frame[t]
                    idx = idx[t]
                else:
                    len_t = min(len(frame), 500)
                    t = np.random.randint(0, len(frame), size=len_t)
                    frame = frame[t]
                    idx = idx[t]
                    # frame = frame[:len_t]
                    # idx = idx[:len_t]
            else:
                raise KeyError
            # 将idx进行归一化
            idx = (idx - s_frame) / (l_frame - s_frame)
            frames.append(frame)
            idxes.append(idx)
        frames = np.array(frames)
        idxes = np.array(idxes)
        # 返回帧，帧位置，标签，name
        return torch.from_numpy(frames).float(), torch.from_numpy(idxes).float(), torch.tensor(label).long(), name


# ####################### add ############################################
# class CrohnSeqFeat(Dataset):

#     def __init__(self, cfg, mode="train"):
#         print('++++>', 'Crohn_seq_feature2 init')
#         self.mode = mode
#         self.cfg = cfg
#         self.batch_size = cfg.TRAIN.BATCH_SIZE
#         self.data = []
#         self.dropout = cfg.TRAIN.DROPOUT
#         try:
#             self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
#         except UnicodeDecodeError:
#             self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV, encoding="gbk")

#         mode_map = {'train': 'Train', 'val': 'Valid', 'test': 'Test'}
#         mode = mode_map[self.mode]

#         for i in range(len(self.df)):
#             datum = self.df.iloc[i]
#             if datum['Fold'] == mode:
#                 name = datum['Name']
#                 if str(name) in os.listdir(self.cfg.DIRS.DATA):
#                     s_frame = datum['S_intestine_frame']
#                     l_frame = datum['L_intestine_frame']
#                     self.data.append((name, datum['Crohn'], s_frame, l_frame))
#         print('len:', len(self.data))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         name, label, s_frame, l_frame = self.data[index]

#         mode = '500'  # '64'
#         pnum = 4  # 32

#         frames = []
#         idxes = []
#         for i in range(pnum):
#             frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_500_idx_{i}_of_4.npy'))
#             idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_500_idx_{i}_of_4.npy')).astype(np.float)
#             # assert len(frame) == len(idx), f"{name}, {i}, {len(frame)}, {len(idx)}, {label}"  # TODO
#             if self.mode == 'train':
#                 len_t = min(len(frame), len(idx))
#                 t = np.random.randint(0, len_t, size=450)
#                 frame = frame[t]
#                 idx = idx[t]
#             else:
#                 frame = frame[:450]
#                 idx = idx[:450]

#             idx = (idx - s_frame) / (l_frame - s_frame)
#             frames.append(frame)
#             idxes.append(idx)

#         for i in range(pnum-1):
#             frame_1 = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_500_idx_{i}_of_4.npy'))
#             idx_1 = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_500_idx_{i}_of_4.npy')).astype(np.float)
#             frame_2 = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_500_idx_{i+1}_of_4.npy'))
#             idx_2 = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_500_idx_{i+1}_of_4.npy')).astype(np.float)
#             if self.mode == 'train':
#                 len_t = min(len(frame_1), len(idx_1))
#                 t_1 = np.random.randint(0, len_t, size=250)
#                 t_2 = np.random.randint(0, len_t, size=200)
#                 frame = np.concatenate((frame_1[t_1],frame_2[t_2]),axis=0)
#                 idx = np.concatenate((idx_1[t_1],idx_2[t_2]),axis=0)
#             else:
#                 frame_1 = frame_1[:250]
#                 frame_2 = frame_2[:200]
#                 idx_1 = idx_1[:250]
#                 idx_2 = idx_2[:200]
#                 frame = np.concatenate((frame_1,frame_2),axis=0)
#                 idx = np.concatenate((idx_1,idx_2),axis=0)

#             idx = (idx - s_frame) / (l_frame - s_frame)
#             frames.append(frame)
#             idxes.append(idx)

#         frames = np.array(frames)
#         idxes = np.array(idxes)

#         return torch.from_numpy(frames).float(), torch.from_numpy(idxes).float(), torch.tensor(label).long(), name

# class CrohnSeqFeat(Dataset):

#     def __init__(self, cfg, mode="train"):
#         print('++++>', 'Crohn_seq_feature2 init')
#         self.mode = mode
#         self.cfg = cfg
#         self.batch_size = cfg.TRAIN.BATCH_SIZE
#         self.data = []
#         self.dropout = cfg.TRAIN.DROPOUT
#         try:
#             self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
#         except UnicodeDecodeError:
#             self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV, encoding="gbk")

#         mode_map = {'train': 'Train', 'val': 'Valid', 'test': 'Test'}
#         mode = mode_map[self.mode]

#         for i in range(len(self.df)):
#             datum = self.df.iloc[i]
#             if datum['Fold'] == mode:
#                 name = datum['Name']
#                 if str(name) in os.listdir(self.cfg.DIRS.DATA):
#                     s_frame = datum['S_intestine_frame']
#                     l_frame = datum['L_intestine_frame']
#                     self.data.append((name, datum['Crohn'], s_frame, l_frame))
#         print('len:', len(self.data))

#     def __len__(self):
#         if self.mode == 'train':
#             return len(self.data) * 2
#         else:
#             return len(self.data)

#     def __getitem__(self, index):
#         if self.mode == 'train':
#             name, label, s_frame, l_frame = self.data[index // 2]
#         else:
#             name, label, s_frame, l_frame = self.data[index]

#         mode = '1000'  # '64'
#         pnum = 4  # 32

#         frames = []
#         idxes = []
#         for i in range(pnum):
#             if mode == '1000':
#                 frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_1000_idx_{i}_of_4.npy'))
#                 idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_1000_idx_{i}_of_4.npy')).astype(np.float)
#                 # assert len(frame) == len(idx), f"{name}, {i}, {len(frame)}, {len(idx)}, {label}"  # TODO
#                 if self.mode == 'train':
#                     if index % 2 == 1:
#                         len_t = min(len(frame), len(idx))
#                         t = np.random.randint(0, len_t, size=400)
#                     else:
#                         t = np.random.randint(0, 500, size=400)
#                     frame = frame[t]
#                     idx = idx[t]
#                 else:
#                     frame = frame[:400]
#                     idx = idx[:400]
#             elif mode == '200':
#                 frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_200_idx_{i}_of_4.npy'))
#                 idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_200_idx_{i}_of_4.npy')).astype(np.float)
#             elif mode == '64':
#                 frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_64_idx_{i}_of_32.npy'))
#                 idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_64_idx_{i}_of_32.npy')).astype(np.float)
#             elif mode == '500':
#                 frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'top_500_idx_{i}_of_4.npy'))
#                 idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx_top_500_idx_{i}_of_4.npy')).astype(np.float)
#                 # assert len(frame) == len(idx), f"{name}, {i}, {len(frame)}, {len(idx)}, {label}"  # TODO
#                 if self.mode == 'train':
#                     len_t = min(len(frame), len(idx))
#                     t = np.random.randint(0, len_t, size=450)
#                     frame = frame[t]
#                     idx = idx[t]
#                 else:
#                     frame = frame[:450]
#                     idx = idx[:450]
#             elif mode == '-1':
#                 frame = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'features.npy'))
#                 idx = np.load(os.path.join(self.cfg.DIRS.DATA, name, f'idx.npy')).astype(np.float)

#                 if self.mode == 'train':
#                     len_t = min(len(frame), 500)
#                     t = np.random.randint(0, len(frame), size=len_t)
#                     frame = frame[t]
#                     idx = idx[t]
#                 else:
#                     len_t = min(len(frame), 500)
#                     t = np.random.randint(0, len(frame), size=len_t)
#                     frame = frame[t]
#                     idx = idx[t]
#                     # frame = frame[:len_t]
#                     # idx = idx[:len_t]
#             else:
#                 raise KeyError
#             idx = (idx - s_frame) / (l_frame - s_frame)
#             frames.append(frame)
#             idxes.append(idx)
#         frames = np.array(frames)
#         idxes = np.array(idxes)

#         return torch.from_numpy(frames).float(), torch.from_numpy(idxes).float(), torch.tensor(label).long(), name
# ####################### add ############################################


def str2int(v_str):
    return int(v_str.split('.')[0])


class CrohnOne(Dataset):

    def __init__(self, cfg, mode="train"):
        print('++++>', 'CrohnOne init')
        self.mode = mode
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = []
        # self.seqs = os.listdir(cfg.DIRS.DATA_SEQ)

        self.transform = A.Compose(
            # [A.CenterCrop(cfg.DATA.SIZE, cfg.DATA.SIZE), A.Normalize(), ToTensorV2()])  # Sysu6
            [A.Resize(256, 256), A.Normalize(), ToTensorV2()]) # Rj

        if '20' in self.cfg.DIRS.DATA:
            for pic in os.listdir(os.path.join(self.cfg.DIRS.DATA)):
                self.data.append(os.path.join(self.cfg.DIRS.DATA, pic))
        else:
            self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
            for i in range(len(self.df)):
                datum = self.df.iloc[i]
                if datum['Crohn'] in [0, 1]:
                    sample_name = str(datum['Name'])
                    flod_name = os.path.join(self.cfg.DIRS.DATA, sample_name.rstrip())
                    # if "2016" in str(sample_name):
                    #     flod_name = os.path.join(self.cfg.DIRS.DATA + '2016', sample_name.rstrip())
                    # elif "2017" in sample_name:
                    #     flod_name = os.path.join(self.cfg.DIRS.DATA + '2017', sample_name.rstrip())
                    # elif "2018" in sample_name:
                    #     flod_name = os.path.join(self.cfg.DIRS.DATA + '2018', sample_name.rstrip())
                    # elif "2019" in sample_name:
                    #     flod_name = os.path.join(self.cfg.DIRS.DATA + '2019', sample_name.rstrip())
                    # elif "2020" in sample_name:
                    #     flod_name = os.path.join(self.cfg.DIRS.DATA + '2020', sample_name.rstrip())
                    # else:
                    #     continue
                        # raise ("Error in datum['Name']")
                    images = os.listdir(flod_name)
                    images = sorted(images, key=str2int)
                    # print(datum['Name'], datum['CN_Name'], len(images), images[0], images[-1])
                    for i, pic in enumerate(images):
                        # if i % (len(images) // 100) == 0:
                        self.data.append(os.path.join(flod_name, pic))

        # ==============================================================
        # if '20' in self.cfg.DIRS.DATA:
        #     for pic in os.listdir(os.path.join(self.cfg.DIRS.DATA)):
        #         self.data.append(os.path.join(self.cfg.DIRS.DATA, pic))
        # else:
        #     self.df = pd.read_csv(self.cfg.DIRS.DATA_CSV)
        #     for i in range(len(self.df)):
        #         datum = self.df.iloc[i]
        #         if datum['Crohn'] in [0, 1, 5]:
        #             sample_name = datum['Name']
        #             label = datum['Crohn']
        #             if str(sample_name + '.npy') in self.seqs:
        #                 seq_2000 = list(np.load(os.path.join(cfg.DIRS.DATA_SEQ, str(sample_name + '.npy'))))
        #                 for ii, j in enumerate(seq_2000):
        #                     if ii % 4 == 0:
        #                         self.data.append((j[1], datum['Crohn']))
        # =================================================================

        print('Length of Data:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # (frame_name, label) = self.data[index]
        frame_name = self.data[index]
        image = cv2.imread(frame_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] == 360:
            image = image[20:-20, 20:-20, :]
        else:
            image = image

        # tranform
        frame = self.transform(image=image)
        frame = frame['image']

        label = -1

        return frame, torch.tensor(label).long(), frame_name


def get_dataset(Mode, cfg):
    if Mode == 'train':
        if cfg.DATA.NAME == "Crohn_frame":
            dts = CrohnFrame(cfg, mode="train")
        elif cfg.DATA.NAME == "Crohn_frame_all":
            dts = CrohnFrameAll(cfg, mode="train")
        elif cfg.DATA.NAME == "Crohn_seq_pred":
            dts = CrohnSeqPred(cfg, mode="train")
        elif cfg.DATA.NAME == "Crohn_seq_feature":
            dts = CrohnSeqFeat(cfg, mode="train")
        else:
            raise KeyError
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size,
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)

    elif Mode == "valid":
        if cfg.DATA.NAME == "Crohn_frame":
            dts = CrohnFrame(cfg, mode="val")
        elif cfg.DATA.NAME == "Crohn_frame_all":
            dts = CrohnFrameAll(cfg, mode="val")
        elif cfg.DATA.NAME == "Crohn_seq_pred":
            dts = CrohnSeqPred(cfg, mode="val")
        elif cfg.DATA.NAME == "Crohn_seq_feature":
            dts = CrohnSeqFeat(cfg, mode="val")
        else:
            raise KeyError
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size,
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    elif Mode == "test":
        if cfg.DATA.NAME == "Crohn_frame":
            dts = CrohnFrame(cfg, mode="test")
        elif cfg.DATA.NAME == "Crohn_frame_all":
            dts = CrohnFrameAll(cfg, mode="test")
        elif cfg.DATA.NAME == "Crohn_seq_pred":
            dts = CrohnSeqPred(cfg, mode="test")
        elif cfg.DATA.NAME == "Crohn_seq_feature_prepare":
            dts = CrohnSeqFeatPre(cfg, mode="test")
        elif cfg.DATA.NAME == "Crohn_seq_feature":
            dts = CrohnSeqFeat(cfg, mode="test") ### 修改
        elif cfg.DATA.NAME == "Crohn_seq":
            dts = CrohnSeq(cfg, mode="test")
        elif cfg.DATA.NAME == "Crohn_one":
            dts = CrohnOne(cfg, mode="test")
        else:
            raise KeyError
        batch_size = cfg.TEST.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size,
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    else:
        raise Exception("ERROR in get_dataset()")

    return dataloader
