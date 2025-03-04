import torch
import torch.nn as nn
from torch.optim import Adam
from PIL import Image
import os
import sys
import json
import random
import time
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as Data
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
from scipy.special import softmax
import csv
warnings.filterwarnings('ignore')
# torch.cuda.set_device(2)

y_true = []
y_scores = []

transform = transforms.Compose([
    # transforms.CenterCrop(560), 
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class small_jpg(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.root))
    
    def __getitem__(self, index):
        jpg_list = os.listdir(self.root)
        image_path = os.path.join(self.root, jpg_list[index])
        # print(image_path)
        image = Image.open(image_path)
        image = self.transform(image)
        return image, image_path




root = "/mnt/minio/node77/liuzheng/Six_30/small_pic/u660c 2016_03_02"
dataset = small_jpg(root, transform)
data_iter = Data.DataLoader(dataset = dataset, shuffle = True, batch_size = 32, num_workers = 8, pin_memory=True)
ckpt = torch.load("/mnt/minio/node77/liuzheng/Crohn2Class/best_sysu_efficientnetb4.pth", "cpu")
model = EfficientNet.from_pretrained('efficientnet-b4', '/mnt/minio/node77/liuzheng/Crohn2Class/efficientnet-b4-6ed6700e.pth')
feature = model._fc.in_features
model._fc = nn.Linear(in_features=feature,out_features=2,bias=True)
model = model.cuda()
model.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
                                  strict=True)


model.eval()
tbar = tqdm(data_iter)

Health = 0
Crohn = 0
Health_num = 0
Crohn_num = 0
Health_list = []
Crohn_list = []
Crohn_csv = "/mnt/minio/node77/liuzheng/Six_30/Health/u660c 2016_03_02/Crohn.csv"
Health_csv = "/mnt/minio/node77/liuzheng/Six_30/Health/u660c 2016_03_02/Crohn.csv"
with open(Health_csv, mode='w', newline='') as file_health:
    writer_health = csv.writer(file_health)

with open(Crohn_csv, mode='w', newline='') as file_crohn:
     writer_crohn = csv.writer(file_crohn)
begin = time.time()  
with torch.no_grad():
    for i, batch in enumerate(tbar):
        images, paths = batch
        images = images.cuda()
        # print(images.device)
        # paths.cuda()
        outputs = model(images)
        for index, pred in enumerate(outputs):
            # print(paths[index])
            pred = pred.clone().cpu().numpy()
            pred = softmax(pred)
            # print(pred)
            pred_top = np.argmax(pred)
            # print(pred_top)
            # if 1 - pred[0] == 0.0 or pred[0] == 0.0:
            #     continue
            # y_true.append(0)
            # y_scores.append(1 - pred[0])
            # y_true.append(1)
            # y_scores.append(pred[0])
            # true_label = 0 if 'Health' in paths[index] else 1
            # y_true.append(true_label)
            # y_scores.append(pred[0])
            if pred_top == 1:
                Health += 1
                Health_list.append(paths[index])
                # with open(Health_csv, mode='w', newline='') as file_health:
                #     writer_health = csv.writer(file_health)
                #     writer_health.writerow([paths[index]])
                # file_health.close()
            else:
                Crohn += 1
                Crohn_list.append(paths[index])
                # with open(Crohn_csv, mode='w', newline='') as file_crohn:
                #     writer_crohn = csv.writer(file_crohn)
                #     writer_crohn.writerow([paths[index]])
                # file_health.close()
                
        # if i % 10 == 0:
        #     print(Crohn / (Health + Crohn))

        # if Crohn >= 6000:
        #     break
end = time.time()
# print("Time: ", end - begin)
# with open(Health_csv, mode='w', newline='') as file_health:
#     writer_health = csv.writer(file_health)
#     for item in Health_list:
#         writer_health.writerow([item])

with open(Crohn_csv, mode='w', newline='') as file_crohn:
    writer_crohn = csv.writer(file_crohn)
    for item in Crohn_list:
        writer_crohn.writerow([item])
        
print("Health:", Health)
print("Crohn:", Crohn)        
print(Crohn / (Health + Crohn))
# df_all = pd.DataFrame({
#                     'y_true' : y_true,
#                     'y_scores' : y_scores
#                 })
# df_all.to_csv('/mnt/minio/node77/liuzheng/vit_train/ROC/rj.csv', index=False)

        
# def predict():
#     model.eval()
#     tbar = tqdm(data_iter)
#     Health = []
#     Crohn = []
#     dir_list = os.listdir(root)
#     for index, img in enumerate(tbar):
#         imgs = imgs.cuda()
#         pred = model(imgs)
#         pred = pred.clone().cpu().numpy()
#         pred_top = np.argmax(pred) 
#         path = root + dir_list[index]
#         if pred_top == 0:
#             Health.append(path)
            
        
