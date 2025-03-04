from tqdm import tqdm
import cv2
from pytorch_toolbelt.inference import tta as pytta
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import time 
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

from .LSTM_ResNet import LSTMResNet, TransformerResNet, ResNet_LSTM, ResNet_LSTM_bd, ResNet_TFE, DenseNet_LSTM_bd, \
    DenseNet_TFE, TFC, TF2, TF1
from .RNN_DenseNet import LSTMDenseNet, TransformerDenseNet, DoNothing
from .mffnet import MFFModel
from .densenet import densenet121
from .vit import ViT
from tensorboardX import SummaryWriter

from .utils import AverageMeter, save_checkpoint
from .metrics import DiceScoreStorer, IoUStorer
import warnings

from .show_cam import show_cam
import matplotlib
import torch.nn.functional as F
import torchvision
import skvideo.io
from scipy.special import softmax
from scipy.signal import savgol_filter
from datasets import get_dataset
from PIL import Image
import timm
from collections import Counter
import csv
import pandas as pd
matplotlib.use('AGG')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_model(cfg):
    print('#######', cfg.TRAIN.MODEL, cfg.TRAIN.NUM_CLASS)
    if cfg.TRAIN.MODEL == 'ResNet3D':
        model = torchvision.models.video.r3d_18(pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'ResNet':
        # print("IN")
        model = torchvision.models.resnet34(pretrained=False)
        if cfg.MODEL.IMAGENET_PRETRAIN:
            pthfile = "/opt/data/private/code07/resnet34-333f7ec4.pth"
            model.load_state_dict(torch.load(pthfile))
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'VGG11':
        model = torchvision.models.vgg11(pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
        fc_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'MobileNet':
        model = torchvision.models.mobilenet_v2(pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
        fc_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'LSTMResNet':
        model = LSTMResNet(cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'LSTMDenseNet':
        model = LSTMDenseNet(cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'ResNet_LSTM':
        model = ResNet_LSTM(cfg, cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'ResNet_LSTM_bd':
        model = ResNet_LSTM_bd(cfg, cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'ResNet_TFE':
        model = ResNet_TFE(cfg, cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'DenseNet_LSTM_bd':
        model = DenseNet_LSTM_bd(cfg, cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'DenseNet_TFE':
        model = DenseNet_TFE(cfg, cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'TransformerDenseNet':
        model = TransformerDenseNet(cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'TransformerResNet':
        model = TransformerResNet(cfg.TRAIN.NUM_CLASS, pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
    elif cfg.TRAIN.MODEL == 'TFC':
        model = TFC(cfg.TRAIN.NUM_CLASS, fc_features=cfg.DATA.INP_CHANNELS)
    elif cfg.TRAIN.MODEL == 'TF2':
        model = TF2(cfg.TRAIN.NUM_CLASS, fc_features=cfg.DATA.INP_CHANNELS)
    elif cfg.TRAIN.MODEL == 'TF1':
        model = TF1(cfg.TRAIN.NUM_CLASS, fc_features=cfg.DATA.INP_CHANNELS)
    elif cfg.TRAIN.MODEL == 'ViT':
        model = ViT(
            image_size=224,
            patch_size=16,
            num_classes=cfg.TRAIN.NUM_CLASS,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif cfg.TRAIN.MODEL == 'ViT_timm':
        model = timm.create_model(cfg.TRAIN.VIT_NAME, pretrained=cfg.MODEL.IMAGENET_PRETRAIN,
                                  num_classes=cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'DenseNet':
        model = torchvision.models.densenet121(pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
        fc_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'densenet':
        model = densenet121(pretrained=cfg.MODEL.IMAGENET_PRETRAIN)
        fc_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    elif cfg.TRAIN.MODEL == 'MFFModel':
        model = MFFModel(cfg=cfg)
    elif cfg.TRAIN.MODEL == 'Efficientnet':
        class EfficientNetB4Custom(nn.Module):
            def __init__(self, num_classes=1000):
                super(EfficientNetB4Custom, self).__init__()
                # 加载 EfficientNet-B4 预训练模型
                self.model = EfficientNet.from_pretrained('efficientnet-b4', '/opt/data/private/EfficientNet/efficientnet-b4-6ed6700e.pth')  # 首先创建模型，不加载权重('efficientnet_b4', pretrained=True)
                # 获取 EfficientNet-B4 分类层前的特征数
                in_features = self.model._fc.in_features  # 应为 1792
                # 新增一个线性层，将特征维数降到 512
                self.additional_fc = nn.Linear(512, num_classes)
                # 原始分类层，将输出调整到指定类别数
                self.model._fc = nn.Linear(in_features, 512)
            def forward(self, x):
                # 使用 EfficientNet 的特征提取部分
                x = self.model(x)
                x = torch.relu(x)  # 使用 ReLU 激活函数
                # 新增的线性层
                x = self.additional_fc(x)
                # 最后的分类层
                return x
        # 使用示例
        model = EfficientNetB4Custom(num_classes=2)
        # model = EfficientNet.from_pretrained('efficientnet-b4', '/opt/data/private/EfficientNet/efficientnet-b4-6ed6700e.pth')  # 首先创建模型，不加载权重
        # feature = model._fc.in_features
        # model._fc = nn.Linear(in_features=feature,out_features=cfg.TRAIN.NUM_CLASS,bias=True)
        # ckpt = torch.load("/opt/data/private/EfficientNet/best_sysu_efficientnetb4.pth", "cpu")
        # model.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
        #                           strict=True)
    else:
        raise Exception("Model not found")

    return model


def test_model_backup(_print, cfg, model, test_loaders, weight=None, tta=False):
    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    if weight is not None:
        print(weight)
        model.load_state_dict(torch.load(weight)["state_dict"])

    state = torch.no_grad()
    if cfg.TEST.CAM is False:
        model.eval()
    else:
        model.train()
        state = torch.enable_grad()

    test_loader, data_prepare = test_loaders
    tbar = tqdm(test_loader)
    ans = [[0, 0, 0]] * 3
    gt_label = [0, 0, 0]
    gt_whole = [0, 0, 0]
    ans = np.array(ans)
    print("test begin")
    os.makedirs(cfg.DIRS.TEST, exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'jpg'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'y'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'feature'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'seg'), exist_ok=True)
    bingo = miss = 0
    frame_num = 10000
    ious = []
    names = []
    with state:
        for batch in tbar:
            if cfg.DATA.NAME in ["Crohn_all", "Crohn_frame"]:
                image, target, name = batch
                frame_num = 1000
            elif cfg.DATA.NAME == "Crohn_frame_andSeg":
                image, target, mask, weight, name = batch
                mask = mask * 255
                # mask = mask.cuda()
                # weight = weight.cuda()
            else:
                avi_name, stomach_time, s_intestine_time, l_intestine_time = batch
                print('==>', avi_name, stomach_time, s_intestine_time, l_intestine_time)
                tmp = cv2.VideoCapture(avi_name[0])
                try:
                    p_name = avi_name[0].split("/")[-2]
                except IndexError:
                    p_name = avi_name[0].split("\\")[-2]
                frame_num = int(tmp.get(7))
                print('======>', frame_num)
            ans_whole = []
            label_whole = []

            # get start and end
            if cfg.TEST.PRED == 1:
                start = frame_num // 2
                end = frame_num // 2

                step = frame_num * 3 // 8
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=start)
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"start{start:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        # ans[int(pred_top)][int(target[i].item())] += 1

                        if pred_top == 0:
                            start += step
                            start = min(start, frame_num - 1)
                        else:
                            start -= step
                            start = max(1, start)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                step = frame_num * 3 // 8
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=end)
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"end{end:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        # ans[int(pred_top)][int(target[i].item())] += 1

                        end_before = start
                        if pred_top == 2:
                            end -= step
                            end = max(1, end)
                        else:
                            end += step
                            end = min(end, frame_num - 1)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                print('===>', start, end, s_intestine_time, l_intestine_time)
                if end < s_intestine_time or start > l_intestine_time:
                    iou = 0
                else:
                    tmp = [start, end, s_intestine_time, l_intestine_time]
                    tmp = np.array(tmp)
                    tmp = np.sort(tmp)
                    iou = (tmp[2] - tmp[1]) / (tmp[3] - tmp[0])
                print(iou)
                ious.append(iou)
                names.append((avi_name, start, end))
                save_root = cfg.DIRS.TEST
                save_name = p_name
                to_txt = np.array([start, end, s_intestine_time, l_intestine_time])
                os.makedirs(os.path.join(save_root, save_name), exist_ok=True)
                np.savetxt(os.path.join(save_root, save_name, f"iou_{iou}.txt"), to_txt, fmt='%d')

            # get start and end
            elif cfg.TEST.PRED == 2:
                start = frame_num // 2
                end = frame_num // 2
                start_a = []
                end_a = []
                n_x1 = 0
                n_x2 = 0

                step = int(frame_num * 3 // 8)
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=start)
                    start_a.append(start)
                    n_x1 += 1
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"start{start:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        pred = softmax(pred)
                        alpha = max(pred[pred_top] - 0.5, 0) + 0.005
                        print(pred, alpha)
                        # ans[int(pred_top)][int(target[i].item())] += 1

                        if pred_top == 0:
                            # print('++++++++++++++++++++++++++++++', start, int(step * 2 * alpha))
                            start += int(step * 2 * alpha)
                            # print('==============================', start, int(step * 2 * alpha))
                            start = min(start, frame_num - step // 10)
                        else:
                            start -= int(step * 2 * alpha)
                            start = max(1 + step // 10, start)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                step = frame_num * 3 // 8
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=end)
                    end_a.append(end)
                    n_x2 += 1
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"end{end:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        pred = softmax(pred)
                        alpha = max(pred[pred_top] - 0.5, 0) + 0.005
                        print(pred, alpha)

                        end_before = start
                        if pred_top == 2:
                            end -= int(step * 2 * alpha)
                            end = max(1 + step // 10, end)
                        else:
                            end += int(step * 2 * alpha)
                            end = min(end, frame_num - step // 10)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                print('===>', start, end, s_intestine_time, l_intestine_time)
                if end < s_intestine_time or start > l_intestine_time:
                    iou = 0
                else:
                    tmp = [start, end, s_intestine_time, l_intestine_time]
                    tmp = np.array(tmp)
                    tmp = np.sort(tmp)
                    iou = (tmp[2] - tmp[1]) / (tmp[3] - tmp[0])
                print(iou)
                ious.append(iou)
                names.append((avi_name, start, end))
                save_root = cfg.DIRS.TEST
                save_name = p_name
                to_txt = np.array([start, end, s_intestine_time, l_intestine_time])
                os.makedirs(os.path.join(save_root, save_name), exist_ok=True)
                np.savetxt(os.path.join(save_root, save_name, f"iou_{iou}.txt"), to_txt, fmt='%d')

                plt.figure()
                plt.plot(list(range(n_x1)), start_a, label="start")
                plt.plot(list(range(n_x2)), end_a, label="end")
                plt.plot(list(range(n_x1)), [s_intestine_time.item()] * n_x1, label="start_gt", alpha=0.3)
                plt.plot(list(range(n_x2)), [l_intestine_time.item()] * n_x2, label="end_gt", alpha=0.3)
                plt.xlabel("iter")
                plt.ylabel("num_frame")
                plt.legend()
                plt.savefig(os.path.join(save_root, save_name, f"01.jpg"))

            # classify some images
            elif cfg.TEST.PRED == 0:
                for index in range(frame_num // 1000):
                    # if index % 100 == 0:
                    #     print(index*1000, frame_num)
                    if cfg.DATA.NAME not in ["Crohn_all", "Crohn_frame", "Crohn_frame_semi"]:  # TODO
                        image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                     i=index * 1000)
                        # image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time)
                    # print(image.size())
                    # print(image[:, :, :, 160, 160])
                    image = image.cuda()
                    # target = target.cuda()
                    output = model(image)
                    pred_sm = F.softmax(output)
                    print("pred_sm, target", pred_sm, target)
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"index{index}", target)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        ans[int(pred_top)][int(target[i].item())] += 1
                        ans_whole.append(int(pred_top))
                        label_whole.append(int(target[i].item()))
                        gt_whole[target[i].item()] += 1
                        if pred_top == target[i].item():
                            bingo += 1
                            gt_label[target[i].item()] += 1
                        else:
                            miss += 1
                # plt.plot(range(frame_num//10000), ans_whole)
                # plt.plot(range(frame_num // 10000), label_whole)
                # plt.savefig(f"{avi_name[0][:-4]}_test_3D.png")
                # plt.close()

            # classify all images in a video one by one
            elif cfg.TEST.PRED == 4:
                videodata = skvideo.io.vread(avi_name[0])
                videodata = videodata.transpose(0, 3, 1, 2)
                b_len = 100
                print("videodata", videodata.shape)
                videodata = videodata[:, [2, 1, 0], :, :]
                ans = []
                ans_sm = []
                for index in range(0, frame_num, b_len):
                    # if index % 100 == 0:
                    print(index, '/', frame_num)
                    # frame1 = None
                    # for i in range(b_len):
                    #     image, _ = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time, i=index+i)
                    #     if i == 0:
                    #         frame1 = image
                    #     else:
                    #         frame1 = np.concatenate((frame1, image), axis=0)

                    frame = videodata[index: min(index + b_len, frame_num)]
                    frame[:, :, 305:, :131] = 0
                    frame[:, :, :15, 260:] = 0

                    # print(frame[:, :, 160, 160], frame1[:, :, 160, 160])

                    image = torch.from_numpy(frame).float()
                    image = image.cuda()
                    # target = target.cuda()
                    output = model(image)
                    # print(output, index)
                    for i, pred in enumerate(output):
                        # print(pred)
                        ans.append(pred.cpu().numpy())
                        pred_sm = F.softmax(pred)
                        ans_sm.append(pred_sm.cpu().numpy())

                np_a = np.array(ans_sm)
                np.save(os.path.join(cfg.DIRS.TEST, f"ans_{avi_name[0].split('/')[-2]}.npy"), np.array(ans))
                np.save(os.path.join(cfg.DIRS.TEST, f"ans_sm_{avi_name[0].split('/')[-2]}.npy"), np_a)
                np.savetxt(os.path.join(cfg.DIRS.TEST, f"ans_sm_{avi_name[0].split('/')[-2]}.csv"), np_a, delimiter=',',
                           fmt='%.4f')

            # classify all images in a video with dataloader
            elif cfg.TEST.PRED == 5:
                video_capture = tmp
                n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                print(n_frames)
                n_frames //= 100
                ys = []
                y_features = []
                pred_si = []

                for frame_idx in tqdm(range(n_frames - 1)):
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=frame_idx * 100)
                    image = image.cuda()
                    y, y_feature = model(image, output_feature=True)
                    ys.append(np.squeeze(y.cpu().numpy()))
                    y_features.append(np.squeeze(y_feature.cpu().numpy()))
                    pred_si.append(float(np.squeeze(y.cpu().numpy())[1]))

                pred_si = np.array(pred_si)
                y_features = np.array(y_features)
                ys = np.array(ys)
                try:
                    avi_name = avi_name[0].split("\\")[-2]
                except IndexError:
                    avi_name = avi_name[0].split("/")[-2]
                plt.figure()
                y = pred_si - 0.5
                y = y * (y > 0)
                y *= 2
                y2 = savgol_filter(y, 31, 3)
                x = np.array(list(range(len(pred_si))))
                # plt.plot(x, y)
                x = x * 100
                y2[y2 > 1] = 1
                y2[y2 < 0] = 0
                # y2 /= 2
                # y2 += 0.5
                plt.plot(x, y2)
                begin, end = s_intestine_time, l_intestine_time
                print(begin, end)
                plt.scatter(begin, -0.03, c="red")
                plt.scatter(end, -0.03, c="red")
                plt.plot([begin, end], [-0.03, -0.03], c="red")
                # plt.title(name[7:-5])
                # plt.ylim(ymin=0.45)
                # plt.ylim(ymax=1.05)
                save_root = cfg.DIRS.TEST
                plt.savefig(os.path.join(save_root, 'jpg', f'temp_{avi_name}.jpg'))

                np.save(os.path.join(save_root, 'y', f'temp_{avi_name}'), ys)
                np.save(os.path.join(save_root, 'feature', f'temp_{avi_name}'), y_features)

            # output MFF CAM
            elif cfg.TEST.PRED == 6:
                BS, C, H, W = image.size()
                # print(BS, C, H, W)
                image = image.cuda()
                preds, preds_seg = model(image, mode="test")  # [0, target, :, :].cpu().numpy()
                preds = preds.mean(dim=(2, 3))

                for i in range(BS):
                    pred = preds[i]
                    pred_seg = preds_seg[i]
                    pred = pred.clone().cpu().numpy()
                    pred_top = np.argmax(pred)
                    ans[int(pred_top)][int(target[i].item())] += 1
                    ans_whole.append(int(pred_top))
                    label_whole.append(int(target[i].item()))
                    gt_whole[target[i].item()] += 1
                    print(name[i], pred_top, target[i])
                    if pred_top == target[i].item():
                        bingo += 1
                        gt_label[target[i].item()] += 1
                    else:
                        miss += 1

                    # print("pred_seg", pred_seg.size(), target[i])
                    pred_seg = pred_seg[target[i], :, :].cpu().numpy()
                    # print('pred1', pred_seg.shape, pred_seg.max(), pred_seg.min(), pred_seg.mean())
                    pred_seg = pred_seg / (pred_seg.max() + 1e-6)
                    pred_seg = cv2.resize(pred_seg, (320, 320), interpolation=cv2.INTER_LINEAR)
                    # print('pred2', pred_seg.shape, pred_seg.max(), pred_seg.min(), pred_seg.mean())

                    ## gaussian
                    weights = pred_seg.copy()
                    weights[np.where(pred_seg < 0.2)] = 0
                    weights[np.where(pred_seg > 0.5)] = 0
                    contours = cv2.findContours(np.uint8(weights > 0) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

                    gaus = np.zeros_like(pred_seg)

                    def gaussian(x, y, ux, uy, sx, sy, sxy):
                        c = -1 / (2 * (1 - sxy ** 2 / sx / sy))
                        dx = (x - ux) ** 2 / sx
                        dy = (y - uy) ** 2 / sy
                        dxy = (x - ux) * (y - uy) * sxy / sx / sy
                        return np.exp(c * (dx - 2 * dxy + dy))

                    for j in range(len(contours)):
                        if cv2.contourArea(contours[j]) < 20:
                            continue
                        weight = np.zeros((320, 320, 3))
                        weight = cv2.drawContours(weight, contours, j, color=(1, 1, 1), thickness=-1, lineType=None,
                                                  hierarchy=None, maxLevel=None, offset=None)
                        weight = weight[:, :, 0] * weights
                        weight = weight / weight.sum()
                        X, Y = np.meshgrid(np.arange(320), np.arange(320))
                        ux, uy = (weight * X).sum(), (weight * Y).sum()
                        sx, sy = (weight * (X - ux) ** 2).sum(), (weight * (Y - uy) ** 2).sum()
                        sxy = (weight * (X - ux) * (Y - uy)).sum()
                        gaus = np.maximum(gaus, gaussian(X, Y, ux, uy, sx / 10, sy / 10, sxy / 10))

                    ## fuse = fore + back + gaussian
                    fore = pred_seg * 255
                    # print('fore', fore.shape, fore.max(), fore.min(), fore.mean())
                    back = np.zeros_like(fore)
                    gaus = gaus * 255
                    # print('gaus', gaus.shape, gaus.max(), gaus.min(), gaus.mean())
                    # print('back', back.shape, back.max(), back.min(), back.mean())
                    fuse = np.stack((fore, gaus, back), axis=-1)
                    # print('fuse1', fuse.shape, fuse.max(), fuse.min(), fuse.mean())
                    fuse = cv2.resize(fuse, (W, H), interpolation=cv2.INTER_LINEAR)
                    # print(i, fuse.shape, image[i].size())
                    fuse = fuse  # + image[i].cpu().numpy().transpose(2, 1, 0)
                    # print('fuse2', fuse.shape, fuse.max(), fuse.min(), fuse.mean())

                    ## save
                    save_root = cfg.DIRS.TEST
                    path = os.path.join(save_root, 'seg', name[i].split('/')[-3], name[i].split('/')[-2])
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(save_root, 'seg', name[i].split('/')[-3], name[i].split('/')[-2],
                                             name[i].split('/')[-1].replace('.jpg', '.png')), np.uint8(fuse))

                    image_one = image[i].cpu().numpy().transpose(1, 2, 0)
                    image_one = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)

                    _, _, width, height = image.shape
                    outs = Image.new('RGB', (width * 4, height * 2))

                    im = Image.fromarray(image_one.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (0, 0))

                    # print(fore.mean())
                    im = Image.fromarray(fore.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width, 0))

                    fore_05 = fore
                    fore_05[fore < 128] = 0
                    # fore_05[fore >= 128] = 1
                    im = Image.fromarray(fore_05.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 2, 0))

                    image_one_c = image_one.copy()
                    th, binary = cv2.threshold(fore_05.astype('uint8'), 0, 255, cv2.THRESH_OTSU)
                    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.drawContours(image_one_c, contours, -1, (0, 255, 0), 3)
                    im = Image.fromarray(image_one_c.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 3, 0))

                    im = Image.fromarray(image_one_c.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 3, 0))

                    im = Image.fromarray(fuse.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width, height))

                    im = Image.fromarray((fuse + image_one).astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (0, height))

                    fuse_05 = fuse.sum(2)
                    fuse_05[fuse_05 < 128] = 0
                    fuse_05[fuse_05 >= 128] = 255
                    im = Image.fromarray(fuse_05.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 2, height))

                    image_one_c = image_one.copy()
                    th, binary = cv2.threshold(fuse_05.astype('uint8'), 0, 255, cv2.THRESH_OTSU)
                    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.drawContours(image_one_c, contours, -1, (0, 255, 0), 3)
                    im = Image.fromarray(image_one_c.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 3, height))

                    if cfg.DATA.NAME == "Crohn_frame_andSeg":
                        im = Image.fromarray(mask[i].numpy().astype(np.uint8)).resize((width, height)).convert('RGB')
                        outs.paste(im, (width * 2, height))

                    outs.save(os.path.join(save_root, 'seg', name[i].split('/')[-3], name[i].split('/')[-2],
                                           'img_' + name[i].split('/')[-1].replace('.jpg', '.png')))

    if cfg.TEST.PRED != 0 and cfg.TEST.PRED != 6:
        ious = np.array(ious)
        for i in range(len(ious)):
            print(names[i])
            print(ious[i])
        print('mean, std =>', ious.mean(), ious.std())
        np.savetxt(os.path.join(cfg.DIRS.TEST, f'ans_{ious.mean()}_{ious.std()}.txt'), ious)
    else:
        acc = bingo / (bingo + miss)
        print(ans, acc)
        print(gt_label)
        print(gt_whole)
        print([a / b for a, b in zip(gt_label, gt_whole)])
        print(np.mean([a / b for a, b in zip(gt_label, gt_whole)]))
        np.savetxt(os.path.join(cfg.DIRS.TEST, f'ans_{acc}.txt'), ans)


def test_model(_print, cfg, model, test_loaders, weight=None, tta=False):
    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    if weight is not None:
        print(weight)
        model.load_state_dict(torch.load(weight)["state_dict"])

    state = torch.no_grad()
    if cfg.TEST.CAM is False:
        model.eval()
    else:
        model.train()
        state = torch.enable_grad()

    test_loader = test_loaders
    tbar = tqdm(test_loader)
    ans = [[0, 0]] * 2
    gt_bingo = [0, 0]
    gt_whole = [0, 0]
    ans = np.array(ans)
    print("test begin")
    os.makedirs(cfg.DIRS.TEST, exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'jpg'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'y'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'feature'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'seg'), exist_ok=True)
    bingo = miss = 0
    frame_num = 10000
    bingo_0 = 0
    bingo_1 = 0
    miss_0 = 0
    miss_1 = 0
    ious = []
    names = []
    stats = {}
    #----------add---------
    true_labels = []
    pred_labels = []
    #---------------------
    #---------ROC---------
    Pred_scores = []
    True_labels = []
    #---------------------
    #--------Time---------
    Time = []
    start_time = 0
    end_time = 0
    #---------------------
    with state:
        for batch in tbar:
            if cfg.DATA.NAME in ["Crohn_all", "Crohn_frame", "Crohn_frame_all", "Crohn_seq_pred"]:
                image, target, name = batch
                frame_num = 1000
            elif cfg.DATA.NAME == "Crohn_frame_andSeg":
                image, target, mask, weight, name = batch
                mask = mask * 255
                # mask = mask.cuda()
                # weight = weight.cuda()
            ####################add##################################
            elif cfg.DATA.NAME in ["Crohn_seq_feature"]:
                # return torch.from_numpy(frames).float(), torch.from_numpy(idxes).float(), torch.tensor(label).long(), name
                image, idxes, target, name = batch
            ####################add##################################
            else:
                avi_name, stomach_time, s_intestine_time, l_intestine_time = batch
                print('==>', avi_name, stomach_time, s_intestine_time, l_intestine_time)
                tmp = cv2.VideoCapture(avi_name[0])
                try:
                    p_name = avi_name[0].split("/")[-2]
                except IndexError:
                    p_name = avi_name[0].split("\\")[-2]
                frame_num = int(tmp.get(7))
                print('======>', frame_num)
            ans_whole = []
            label_whole = []

            # classify some images
            if cfg.TEST.PRED == 0:
                for index in range(frame_num // 1000):
                    image = image.to(device='cuda', dtype=torch.float)
                    output = model(image)
                    pred_sm = F.softmax(output)
                    # print("pred_sm, target", pred_sm, target)
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"index{index}", target)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        ans[int(pred_top)][int(target[i].item())] += 1
                        ans_whole.append(int(pred_top))
                        label_whole.append(int(target[i].item()))
                        gt_whole[target[i].item()] += 1
                        if pred_top == target[i].item():
                            bingo += 1
                            gt_bingo[target[i].item()] += 1
                            if target[i].item() == 0:
                                bingo_0 += 1
                            elif target[i].item() == 1:
                                bingo_1 += 1
                        else:
                            miss += 1
                            if target[i].item() == 0:
                                miss_0 += 1
                            elif target[i].item() == 1:
                                miss_1 += 1
                # plt.plot(range(frame_num//10000), ans_whole)
                # plt.plot(range(frame_num // 10000), label_whole)
                # plt.savefig(f"{avi_name[0][:-4]}_test_3D.png")
                # plt.close()

            # get start and end
            elif cfg.TEST.PRED == 1:
                start = frame_num // 2
                end = frame_num // 2

                step = frame_num * 3 // 8
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=start)
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"start{start:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        # ans[int(pred_top)][int(target[i].item())] += 1

                        if pred_top == 0:
                            start += step
                            start = min(start, frame_num - 1)
                        else:
                            start -= step
                            start = max(1, start)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                step = frame_num * 3 // 8
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=end)
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"end{end:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        # ans[int(pred_top)][int(target[i].item())] += 1

                        end_before = start
                        if pred_top == 2:
                            end -= step
                            end = max(1, end)
                        else:
                            end += step
                            end = min(end, frame_num - 1)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                print('===>', start, end, s_intestine_time, l_intestine_time)
                if end < s_intestine_time or start > l_intestine_time:
                    iou = 0
                else:
                    tmp = [start, end, s_intestine_time, l_intestine_time]
                    tmp = np.array(tmp)
                    tmp = np.sort(tmp)
                    iou = (tmp[2] - tmp[1]) / (tmp[3] - tmp[0])
                print(iou)
                ious.append(iou)
                names.append((avi_name, start, end))
                save_root = cfg.DIRS.TEST
                save_name = p_name
                to_txt = np.array([start, end, s_intestine_time, l_intestine_time])
                os.makedirs(os.path.join(save_root, save_name), exist_ok=True)
                np.savetxt(os.path.join(save_root, save_name, f"iou_{iou}.txt"), to_txt, fmt='%d')

            # get start and end
            elif cfg.TEST.PRED == 2:
                start = frame_num // 2
                end = frame_num // 2
                start_a = []
                end_a = []
                n_x1 = 0
                n_x2 = 0

                step = int(frame_num * 3 // 8)
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=start)
                    start_a.append(start)
                    n_x1 += 1
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"start{start:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        pred = softmax(pred)
                        alpha = max(pred[pred_top] - 0.5, 0) + 0.005
                        print(pred, alpha)
                        # ans[int(pred_top)][int(target[i].item())] += 1

                        if pred_top == 0:
                            # print('++++++++++++++++++++++++++++++', start, int(step * 2 * alpha))
                            start += int(step * 2 * alpha)
                            # print('==============================', start, int(step * 2 * alpha))
                            start = min(start, frame_num - step // 10)
                        else:
                            start -= int(step * 2 * alpha)
                            start = max(1 + step // 10, start)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                step = frame_num * 3 // 8
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=end)
                    end_a.append(end)
                    n_x2 += 1
                    image = image.cuda()
                    if cfg.TEST.CAM is True:
                        output = show_cam(cfg, model, image, p_name, f"end{end:05}_step{step:05}", target)
                    else:
                        output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        pred = softmax(pred)
                        alpha = max(pred[pred_top] - 0.5, 0) + 0.005
                        print(pred, alpha)

                        end_before = start
                        if pred_top == 2:
                            end -= int(step * 2 * alpha)
                            end = max(1 + step // 10, end)
                        else:
                            end += int(step * 2 * alpha)
                            end = min(end, frame_num - step // 10)

                        step = step * 9 // 10
                    print('step, start, end', step, start, end)

                print('===>', start, end, s_intestine_time, l_intestine_time)
                if end < s_intestine_time or start > l_intestine_time:
                    iou = 0
                else:
                    tmp = [start, end, s_intestine_time, l_intestine_time]
                    tmp = np.array(tmp)
                    tmp = np.sort(tmp)
                    iou = (tmp[2] - tmp[1]) / (tmp[3] - tmp[0])
                print(iou)
                ious.append(iou)
                names.append((avi_name, start, end))
                save_root = cfg.DIRS.TEST
                save_name = p_name
                to_txt = np.array([start, end, s_intestine_time, l_intestine_time])
                os.makedirs(os.path.join(save_root, save_name), exist_ok=True)
                np.savetxt(os.path.join(save_root, save_name, f"iou_{iou}.txt"), to_txt, fmt='%d')

                plt.figure()
                plt.plot(list(range(n_x1)), start_a, label="start")
                plt.plot(list(range(n_x2)), end_a, label="end")
                plt.plot(list(range(n_x1)), [s_intestine_time.item()] * n_x1, label="start_gt", alpha=0.3)
                plt.plot(list(range(n_x2)), [l_intestine_time.item()] * n_x2, label="end_gt", alpha=0.3)
                plt.xlabel("iter")
                plt.ylabel("num_frame")
                plt.legend()
                plt.savefig(os.path.join(save_root, save_name, f"01.jpg"))

            # classify images and statistics
            elif cfg.TEST.PRED == 3:
                image = image.to(device='cuda', dtype=torch.float)
                output = model(image)
                for i, pred in enumerate(output):
                    n = name[i].split('/')[-2] + '_' + name[i].split('/')[-3]
                    if n not in stats:
                        stats[n] = [0, 0]
                    pred = pred.clone().cpu().numpy()
                    pred_top = np.argmax(pred)
                    ans[int(pred_top)][int(target[i].item())] += 1
                    ans_whole.append(int(pred_top))
                    label_whole.append(int(target[i].item()))
                    gt_whole[target[i].item()] += 1

                    if pred_top == target[i].item():
                        stats[n][0] += 1
                        bingo += 1
                        gt_bingo[target[i].item()] += 1
                        if target[i].item() == 0:
                            bingo_0 += 1
                        elif target[i].item() == 1:
                            bingo_1 += 1
                    else:
                        stats[n][1] += 1
                        miss += 1
                        if target[i].item() == 0:
                            miss_0 += 1
                        elif target[i].item() == 1:
                            miss_1 += 1

            # classify all images in a video one by one
            elif cfg.TEST.PRED == 4:
                videodata = skvideo.io.vread(avi_name[0])
                videodata = videodata.transpose(0, 3, 1, 2)
                b_len = 100
                print("videodata", videodata.shape)
                videodata = videodata[:, [2, 1, 0], :, :]
                ans = []
                ans_sm = []
                for index in range(0, frame_num, b_len):
                    # if index % 100 == 0:
                    print(index, '/', frame_num)
                    # frame1 = None
                    # for i in range(b_len):
                    #     image, _ = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time, i=index+i)
                    #     if i == 0:
                    #         frame1 = image
                    #     else:
                    #         frame1 = np.concatenate((frame1, image), axis=0)

                    frame = videodata[index: min(index + b_len, frame_num)]
                    frame[:, :, 305:, :131] = 0
                    frame[:, :, :15, 260:] = 0

                    # print(frame[:, :, 160, 160], frame1[:, :, 160, 160])

                    image = torch.from_numpy(frame).float()
                    image = image.cuda()
                    # target = target.cuda()
                    output = model(image)
                    # print(output, index)
                    for i, pred in enumerate(output):
                        # print(pred)
                        ans.append(pred.cpu().numpy())
                        pred_sm = F.softmax(pred)
                        ans_sm.append(pred_sm.cpu().numpy())

                np_a = np.array(ans_sm)
                np.save(os.path.join(cfg.DIRS.TEST, f"ans_{avi_name[0].split('/')[-2]}.npy"), np.array(ans))
                np.save(os.path.join(cfg.DIRS.TEST, f"ans_sm_{avi_name[0].split('/')[-2]}.npy"), np_a)
                np.savetxt(os.path.join(cfg.DIRS.TEST, f"ans_sm_{avi_name[0].split('/')[-2]}.csv"), np_a, delimiter=',',
                           fmt='%.4f')

            # classify all images in a video with dataloader
            elif cfg.TEST.PRED == 5:
                video_capture = tmp
                n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                print(n_frames)
                n_frames //= 100
                ys = []
                y_features = []
                pred_si = []

                for frame_idx in tqdm(range(n_frames - 1)):
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=frame_idx * 100)
                    image = image.cuda()
                    y, y_feature = model(image, output_feature=True)
                    ys.append(np.squeeze(y.cpu().numpy()))
                    y_features.append(np.squeeze(y_feature.cpu().numpy()))
                    pred_si.append(float(np.squeeze(y.cpu().numpy())[1]))

                pred_si = np.array(pred_si)
                y_features = np.array(y_features)
                ys = np.array(ys)
                try:
                    avi_name = avi_name[0].split("\\")[-2]
                except IndexError:
                    avi_name = avi_name[0].split("/")[-2]
                plt.figure()
                y = pred_si - 0.5
                y = y * (y > 0)
                y *= 2
                y2 = savgol_filter(y, 31, 3)
                x = np.array(list(range(len(pred_si))))
                # plt.plot(x, y)
                x = x * 100
                y2[y2 > 1] = 1
                y2[y2 < 0] = 0
                # y2 /= 2
                # y2 += 0.5
                plt.plot(x, y2)
                begin, end = s_intestine_time, l_intestine_time
                print(begin, end)
                plt.scatter(begin, -0.03, c="red")
                plt.scatter(end, -0.03, c="red")
                plt.plot([begin, end], [-0.03, -0.03], c="red")
                # plt.title(name[7:-5])
                # plt.ylim(ymin=0.45)
                # plt.ylim(ymax=1.05)
                save_root = cfg.DIRS.TEST
                plt.savefig(os.path.join(save_root, 'jpg', f'temp_{avi_name}.jpg'))

                np.save(os.path.join(save_root, 'y', f'temp_{avi_name}'), ys)
                np.save(os.path.join(save_root, 'feature', f'temp_{avi_name}'), y_features)

            # output MFF CAM
            elif cfg.TEST.PRED == 6:
                BS, C, H, W = image.size()
                # print(BS, C, H, W)
                image = image.cuda()
                preds, preds_seg = model(image, mode="test")  # [0, target, :, :].cpu().numpy()
                preds = preds.mean(dim=(2, 3))

                for i in range(BS):
                    pred = preds[i]
                    pred_seg = preds_seg[i]
                    pred = pred.clone().cpu().numpy()
                    pred_top = np.argmax(pred)
                    ans[int(pred_top)][int(target[i].item())] += 1
                    ans_whole.append(int(pred_top))
                    label_whole.append(int(target[i].item()))
                    gt_whole[target[i].item()] += 1
                    print(name[i], pred_top, target[i])
                    if pred_top == target[i].item():
                        bingo += 1
                        gt_bingo[target[i].item()] += 1
                    else:
                        miss += 1

                    # print("pred_seg", pred_seg.size(), target[i])
                    pred_seg = pred_seg[target[i], :, :].cpu().numpy()
                    # print('pred1', pred_seg.shape, pred_seg.max(), pred_seg.min(), pred_seg.mean())
                    pred_seg = pred_seg / (pred_seg.max() + 1e-6)
                    pred_seg = cv2.resize(pred_seg, (320, 320), interpolation=cv2.INTER_LINEAR)
                    # print('pred2', pred_seg.shape, pred_seg.max(), pred_seg.min(), pred_seg.mean())

                    ## gaussian
                    weights = pred_seg.copy()
                    weights[np.where(pred_seg < 0.2)] = 0
                    weights[np.where(pred_seg > 0.5)] = 0
                    contours = cv2.findContours(np.uint8(weights > 0) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

                    gaus = np.zeros_like(pred_seg)

                    def gaussian(x, y, ux, uy, sx, sy, sxy):
                        c = -1 / (2 * (1 - sxy ** 2 / sx / sy))
                        dx = (x - ux) ** 2 / sx
                        dy = (y - uy) ** 2 / sy
                        dxy = (x - ux) * (y - uy) * sxy / sx / sy
                        return np.exp(c * (dx - 2 * dxy + dy))

                    for j in range(len(contours)):
                        if cv2.contourArea(contours[j]) < 20:
                            continue
                        weight = np.zeros((320, 320, 3))
                        weight = cv2.drawContours(weight, contours, j, color=(1, 1, 1), thickness=-1, lineType=None,
                                                  hierarchy=None, maxLevel=None, offset=None)
                        weight = weight[:, :, 0] * weights
                        weight = weight / weight.sum()
                        X, Y = np.meshgrid(np.arange(320), np.arange(320))
                        ux, uy = (weight * X).sum(), (weight * Y).sum()
                        sx, sy = (weight * (X - ux) ** 2).sum(), (weight * (Y - uy) ** 2).sum()
                        sxy = (weight * (X - ux) * (Y - uy)).sum()
                        gaus = np.maximum(gaus, gaussian(X, Y, ux, uy, sx / 10, sy / 10, sxy / 10))

                    ## fuse = fore + back + gaussian
                    fore = pred_seg * 255
                    # print('fore', fore.shape, fore.max(), fore.min(), fore.mean())
                    back = np.zeros_like(fore)
                    gaus = gaus * 255
                    # print('gaus', gaus.shape, gaus.max(), gaus.min(), gaus.mean())
                    # print('back', back.shape, back.max(), back.min(), back.mean())
                    fuse = np.stack((fore, gaus, back), axis=-1)
                    # print('fuse1', fuse.shape, fuse.max(), fuse.min(), fuse.mean())
                    fuse = cv2.resize(fuse, (W, H), interpolation=cv2.INTER_LINEAR)
                    # print(i, fuse.shape, image[i].size())
                    fuse = fuse  # + image[i].cpu().numpy().transpose(2, 1, 0)
                    # print('fuse2', fuse.shape, fuse.max(), fuse.min(), fuse.mean())

                    ## save
                    save_root = cfg.DIRS.TEST
                    path = os.path.join(save_root, 'seg', name[i].split('/')[-3], name[i].split('/')[-2])
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(save_root, 'seg', name[i].split('/')[-3], name[i].split('/')[-2],
                                             name[i].split('/')[-1].replace('.jpg', '.png')), np.uint8(fuse))

                    image_one = image[i].cpu().numpy().transpose(1, 2, 0)
                    image_one = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)

                    _, _, width, height = image.shape
                    outs = Image.new('RGB', (width * 4, height * 2))

                    im = Image.fromarray(image_one.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (0, 0))

                    # print(fore.mean())
                    im = Image.fromarray(fore.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width, 0))

                    fore_05 = fore
                    fore_05[fore < 128] = 0
                    # fore_05[fore >= 128] = 1
                    im = Image.fromarray(fore_05.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 2, 0))

                    image_one_c = image_one.copy()
                    th, binary = cv2.threshold(fore_05.astype('uint8'), 0, 255, cv2.THRESH_OTSU)
                    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.drawContours(image_one_c, contours, -1, (0, 255, 0), 3)
                    im = Image.fromarray(image_one_c.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 3, 0))

                    im = Image.fromarray(image_one_c.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 3, 0))

                    im = Image.fromarray(fuse.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width, height))

                    im = Image.fromarray((fuse + image_one).astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (0, height))

                    fuse_05 = fuse.sum(2)
                    fuse_05[fuse_05 < 128] = 0
                    fuse_05[fuse_05 >= 128] = 255
                    im = Image.fromarray(fuse_05.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 2, height))

                    image_one_c = image_one.copy()
                    th, binary = cv2.threshold(fuse_05.astype('uint8'), 0, 255, cv2.THRESH_OTSU)
                    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.drawContours(image_one_c, contours, -1, (0, 255, 0), 3)
                    im = Image.fromarray(image_one_c.astype(np.uint8)).resize((width, height)).convert('RGB')
                    outs.paste(im, (width * 3, height))

                    if cfg.DATA.NAME == "Crohn_frame_andSeg":
                        im = Image.fromarray(mask[i].numpy().astype(np.uint8)).resize((width, height)).convert('RGB')
                        outs.paste(im, (width * 2, height))

                    outs.save(os.path.join(save_root, 'seg', name[i].split('/')[-3], name[i].split('/')[-2],
                                           'img_' + name[i].split('/')[-1].replace('.jpg', '.png')))

            # predict crohn (average 4 results)
            elif cfg.TEST.PRED == 7:
                # if cfg.DATA.NAME in ["Crohn_frame", "Crohn_seq_pred", "Crohn_frame_all"]:
                #     image, target, name = batch
                # else:
                #     raise KeyError

                image1 = image[0].to(device='cuda', dtype=torch.float)
                output1 = torch.softmax(model(image1), 1)
                image2 = image[1].to(device='cuda', dtype=torch.float)
                output2 = torch.softmax(model(image2), 1)
                image3 = image[2].to(device='cuda', dtype=torch.float)
                output3 = torch.softmax(model(image3), 1)
                image4 = image[3].to(device='cuda', dtype=torch.float)
                output4 = torch.softmax(model(image4), 1)
                target = target.cuda()
                output = (output1 + output2 + output3 + output4) / 4.0

                for i, pred in enumerate(output):
                    pred = pred.clone().cpu().numpy()
                    pred_top = np.argmax(pred)
                    preds = [output1[i].cpu().numpy(), output2[i].cpu().numpy(), output3[i].cpu().numpy(),
                             output4[i].cpu().numpy()]
                    preds_top = np.argmax(preds, 1)
                    # print(preds_top, int(target[i].item()))
                    ans[int(pred_top)][int(target[i].item())] += 1
                    ans_whole.append(int(pred_top))
                    label_whole.append(int(target[i].item()))
                    gt_whole[target[i].item()] += 1
                    if pred_top == target[i].item():
                        bingo += 1
                        gt_bingo[target[i].item()] += 1
                        if target[i].item() == 0:
                            bingo_0 += 1
                        elif target[i].item() == 1:
                            bingo_1 += 1
                    else:
                        print(preds_top, int(target[i].item()))
                        miss += 1
                        if target[i].item() == 0:
                            miss_0 += 1
                        elif target[i].item() == 1:
                            miss_1 += 1

            # just like validation
            elif cfg.TEST.PRED == 8:
                # if cfg.DATA.NAME in ["Crohn_frame", "Crohn_seq_pred", "Crohn_frame_all"]:
                #     image, target, name = batch
                # else:
                #     raise KeyError

                image1 = image[0].to(device='cuda', dtype=torch.float)
                output1 = torch.softmax(model(image1), 1)
                image2 = image[1].to(device='cuda', dtype=torch.float)
                output2 = torch.softmax(model(image2), 1)
                image3 = image[2].to(device='cuda', dtype=torch.float)
                output3 = torch.softmax(model(image3), 1)
                image4 = image[3].to(device='cuda', dtype=torch.float)
                output4 = torch.softmax(model(image4), 1)
                target = target.cuda()
                output = (output1 + output2 + output3 + output4) / 4.0

                for i, pred in enumerate(output):
                    pred = pred.clone().cpu().numpy()
                    pred_top = np.argmax(pred)
                    preds = [output1[i].cpu().numpy(), output2[i].cpu().numpy(), output3[i].cpu().numpy(),
                             output4[i].cpu().numpy()]
                    preds_top = np.argmax(preds, 1)
                    # print(preds_top, int(target[i].item()))
                    ans[int(pred_top)][int(target[i].item())] += 1
                    ans_whole.append(int(pred_top))
                    label_whole.append(int(target[i].item()))
                    gt_whole[target[i].item()] += 1
                    if pred_top == target[i].item():
                        bingo += 1
                        gt_bingo[target[i].item()] += 1
                        if target[i].item() == 0:
                            bingo_0 += 1
                        elif target[i].item() == 1:
                            bingo_1 += 1
                    else:
                        print(preds_top, int(target[i].item()))
                        miss += 1
                        if target[i].item() == 0:
                            miss_0 += 1
                        elif target[i].item() == 1:
                            miss_1 += 1
            # ##########################add#################################
            elif cfg.TEST.PRED == 9:
                start_time = time.time()
                image = image.to(device='cuda', dtype=torch.float)
                target = target.cuda()
                idxes = idxes.to(device='cuda', dtype=torch.float)
                output = model(image, idxes)
                for i, pred in enumerate(output):
                    pred = torch.softmax(pred, 0)
                    pred = pred.clone().cpu().numpy()
                    pred_top = np.argmax(pred)
                    ans[int(pred_top)][int(target[i].item())] += 1
                    ans_whole.append(int(pred_top))
                    label_whole.append(int(target[i].item()))
                    gt_whole[target[i].item()] += 1
                    #------------add--------------------
                    pred_labels.append(pred[1])
                    true_labels.append(target[i].item())
                    #---------------------------------
                    #-------------addROC--------------
                    # 获取正类的预测概率 (假设索引1为正类)
                    positive_prob = pred[1]
                    
                    # 记录每个样本的正类预测概率和真实标签
                    Pred_scores.append(positive_prob)
                    True_labels.append(int(target[i].item()))
                    #---------------------------------
                    if pred_top == target[i].item():
                        bingo += 1
                        gt_bingo[target[i].item()] += 1
                        if target[i].item() == 0:
                            bingo_0 += 1
                        elif target[i].item() == 1:
                            bingo_1 += 1
                    else:
                        print(pred_top, int(target[i].item()))
                        miss += 1
                        if target[i].item() == 0: # target = 0 pred = 1
                            miss_0 += 1
                            print(f"miss no crohn: {name[i].split('/')[-1]}")
                            _print(f"miss no crohn: {name[i].split('/')[-1]}")
                        elif target[i].item() == 1: # target = 1 pred = 0
                            miss_1 += 1
                            print(f"miss crohn: {name[i].split('/')[-1]}")
                            _print(f"miss crohn: {name[i].split('/')[-1]}")
            # ##########################add#################################
                end_time = time.time()
                one_time = end_time - start_time
                Time.append(one_time)
    if cfg.TEST.PRED not in [0, 3, 6, 7, 9]:
        ious = np.array(ious)
        for i in range(len(ious)):
            print(names[i])
            print(ious[i])
        print('mean, std =>', ious.mean(), ious.std())
        np.savetxt(os.path.join(cfg.DIRS.TEST, f'ans_{ious.mean()}_{ious.std()}.txt'), ious)
    else:
        acc = bingo / (bingo + miss)
        if bingo_1 + miss_0 != 0:
            P = bingo_1 / (bingo_1 + miss_0)
        else:
            P = 0
        R = bingo_1 / (bingo_1 + miss_1)
        if P + R == 0:
            f1 = 0
        else:
            f1 = 2 * P * R / (P + R)
        _print(f"{ans, acc, f1}")
        _print(f"{gt_bingo}")
        _print(f"{gt_whole}")
        _print([a / b for a, b in zip(gt_bingo, gt_whole)])
        _print(np.mean([a / b for a, b in zip(gt_bingo, gt_whole)]))
        # print(stats)
        if cfg.TEST.PRED == 3:
            for stat in stats:
                _print(f'{stat}, {stats[stat]}')
            stats = sorted(stats.items(), key=lambda kv: (kv[1][1], kv[1][0]))
            # stats = sorted(stats)
            # print(stats)
            _print("After sort:")
            for stat in stats:
                _print(f'{stat}')
        np.savetxt(os.path.join(cfg.DIRS.TEST, f'ans_{cfg.EXP}_{acc}.txt'), ans)
    #---------------------add--------------------------------------------
    if cfg.TEST.PRED == 9:
        auc = roc_auc_score(true_labels, pred_labels)
        _print(f'auc: {auc}')

        data = pd.DataFrame({'Score': Pred_scores, 'Label': True_labels})

        # Specify the path to save the CSV file
        csv_path = "/opt/data/private/senfig/six.csv"

        # Write to CSV
        data.to_csv(csv_path, index=False)

        print(f"Data successfully saved to {csv_path}")

        # ---------Time--------
        # print(Time)
        # print("Time mean:", np.mean(Time))
        # print("Time std", np.std(Time))
        # ---------------------
    #--------------------------------------------------------------------


def valid_model(_print, cfg, model, valid_criterion, valid_loaders, tta=False):
    losses = AverageMeter()
    top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
    top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    model.eval()
    valid_loader = valid_loaders
    tbar = tqdm(valid_loader)

    target_mean_all = []
    pred_mean_all = []

    bingo = 0
    miss = 0
    bingo_0 = bingo_1 = 0
    miss_0 = miss_1 = 0

    gts = []
    preds = []
    # names = {}
    df = pd.read_csv(cfg.DIRS.DATA_CSV)
    # for i in range(len(df)):
    #     datum = df.iloc[i]
    #     names[datum['Name']] = datum['Name']

    with torch.no_grad():
        for i, batch in enumerate(tbar):
            if cfg.DATA.NPY is True:
                image, target = batch
            elif cfg.DATA.NAME in ["Crohn_all", "Crohn_frame_semi"]:
                image, target = batch
            elif cfg.DATA.NAME in ["Crohn_frame", "Crohn_seq_pred", "Crohn_frame_all"]:
                image, target, name = batch
            elif cfg.DATA.NAME in ["Crohn_seq_feature"]:
                image, idxes, target, name = batch
            elif cfg.DATA.NAME == "Crohn_frame_andSeg":
                image, target, mask, weight, name = batch
                mask = mask.cuda()
                weight = weight.cuda()
            else:
                raise KeyError
            # print(image.size(), target.size())
            # print(_id, finding)
            # print(image.max(), image.min(), image.mean(), target.max(), target.min(), target.mean())
            image = image.to(device='cuda', dtype=torch.float)
            target = target.cuda()
            # output = model(image)
            if cfg.DATA.NAME in ["Crohn_seq_feature"]:
                idxes = idxes.to(device='cuda', dtype=torch.float)
                output = model(image, idxes)
            else:
                output = model(image)
            # output = model(image, mode="val")
            # print(output, target)
            if "LSTM" in cfg.TRAIN.MODEL or "MFF" in cfg.TRAIN.MODEL:
                output, output_target2 = output
                output = output.mean(dim=(2, 3))

            # loss
            loss = valid_criterion(output, target)

            if "MFF" in cfg.TRAIN.MODEL:
                if cfg.DATA.NAME == "Crohn_frame_andSeg":
                    B, H, W = mask.size()
                    pred = F.interpolate(output_target2, (H, W), mode='bilinear', align_corners=True)[:, 0, :, :]
                    loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
                else:
                    output_target2 = output_target2.mean(dim=(2, 3))
                    loss2 = -(torch.softmax(output.detach(), dim=1) * torch.log_softmax(output_target2,
                                                                                        dim=1)).sum(dim=1).mean()
                    # print(loss, loss2)
                    loss = 0.5 * (loss + loss2)

            for i, pred in enumerate(output):
                # print(pred)
                pred = torch.softmax(pred, 0)
                # gts.append(target[i].item())
                preds.append(pred[1].item())
                # print(pred)
                pred = pred.clone().cpu().numpy()
                pred_top = np.argmax(pred)
                if pred_top == target[i].item():
                    bingo += 1
                    if target[i].item() == 0:
                        bingo_0 += 1
                    elif target[i].item() == 1:
                        bingo_1 += 1
                else:
                    miss += 1
                    if target[i].item() == 0:
                        # gts.append(names[name[i]])
                        gts.append(name[i].split('/')[-1])
                    if target[i].item() == 0:
                        miss_0 += 1
                    elif target[i].item() == 1:
                        miss_1 += 1

            output = output.max(axis=1)[1]

            # record metrics
            # top_dice.update(output, target)
            # top_iou.update(output, target)
            target_mean = target.type(torch.float).mean()
            pred_mean = output.type(torch.float).mean()

            target_mean_all.append(target_mean.item())
            pred_mean_all.append(pred_mean.item())

            # record
            losses.update(loss.item(), image.size(0))

    print(bingo_0, miss_1)
    print(miss_0, bingo_1)
    acc = bingo / (bingo + miss)
    if bingo_0 + miss_0 == 0:
        acc_0 = 0
    else:
        acc_0 = bingo_0 / (bingo_0 + miss_0)
    acc_1 = bingo_1 / (bingo_1 + miss_1)
    if bingo_1 + miss_0 != 0:
        P = bingo_1 / (bingo_1 + miss_0)
    else:
        P = 0
    R = bingo_1 / (bingo_1 + miss_1)
    if P + R == 0:
        f1 = 0
    else:
        f1 = 2 * P * R / (P + R)
    # _print("Valid iou: %.3f, dice: %.3f loss: %.3f" % (top_iou.avg, top_dice.avg, losses.avg))
    _print("Target mean: %.3f, Pred mean: %.3f, acc: %.3f %.3f %.3f f1: %.3f loss: %.3f" % (
        np.mean(target_mean_all), np.mean(pred_mean_all), acc, acc_0, acc_1, f1, losses.avg))
    print(gts)
    for i in gts:
        print(i)
    print(preds)

    return losses.avg, acc, acc_0, acc_1


def train_loop(_print, cfg, model, train_loader, valid_loader, criterion, valid_criterion, optimizer, scheduler,
               start_epoch, best_metric, test_loaders):
    if cfg.DEBUG == False:
        tb = SummaryWriter(f"./runs/{cfg.EXP}/{cfg.TRAIN.MODEL}",
                           comment=f"{cfg.COMMENT}")  # for visualization
    # weight = cfg.MODEL.WEIGHT
    # model.load_state_dict(torch.load(weight)["state_dict"], strict=False)
    best_acc = 0
    best_acc0 = 0
    best_acc1 = 0
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"Epoch {epoch + 1}")

        # define some meters
        losses = AverageMeter()

        # top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
        # top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
        if cfg.DATA.NAME in ["Crohn_frame_semi"]:
            train_loader = get_dataset('train', cfg)
            # valid_loader = get_dataset('valid', cfg)
        """
        TEST
        """
        # if cfg.TEST.TIMES and epoch % (cfg.TRAIN.EPOCHS // cfg.TEST.TIMES) == 0:
        if epoch != 0:
            cfg_epoch = cfg.clone()
            cfg_epoch.defrost()
            cfg_epoch.DIRS.TEST = os.path.join(cfg_epoch.DIRS.TEST, "test_epoch" + str(epoch))
            test_model(_print, cfg_epoch, model, test_loaders, weight=None, tta=cfg.INFER.TTA)
            # test_model_one(logging.info, cfg, model, test_loaders, weight=cfg.MODEL.WEIGHT)  # for test02_gettop2000(select top2000frames)
            # test_model_one_backup(logging.info, cfg, model, test_loaders, weight=cfg.MODEL.WEIGHT)  # for test01_getframesnpy
            # cfg_epoch.DIRS.TEST = os.path.join(cfg_epoch.DIRS.TEST, "train_epoch" + str(epoch))
            # test_model(_print, cfg_epoch, model, train_loaders, weight=None, tta=cfg.INFER.TTA)

        """
        TRAINING
        """
        # switch model to training mode
        model.train()

        train_loader = train_loader
        tbar = tqdm(train_loader)

        target_mean_all = []
        pred_mean_all = []

        for i, batch in enumerate(tbar):
            criterion_res = criterion
            if cfg.DATA.NAME in ["Crohn_all", "Crohn_frame_semi"]:
                image, target = batch
            elif cfg.DATA.NAME in ["Crohn_frame", "Crohn_seq_pred", "Crohn_frame_all"]:
                image, target, name = batch
            elif cfg.DATA.NAME in ["Crohn_seq_feature"]:
                image, idxes, target, name = batch
            elif cfg.DATA.NAME == "Crohn_frame_andSeg":
                image, target, mask, weight, name = batch
                mask = mask.cuda()
                weight = weight.cuda()
            else:
                raise KeyError

            # print(image.size(), target.size())
            # print('image[100, 100]', image[0, :, 100, 100])

            image = image.to(device='cuda', dtype=torch.float)
            target = target.cuda()

            if cfg.DATA.NAME in ["Crohn_seq_feature"]:
                idxes = idxes.to(device='cuda', dtype=torch.float)
                output_target = model(image, idxes)
            else:
                output_target = model(image)
            if "LSTM" in cfg.TRAIN.MODEL or "MFF" in cfg.TRAIN.MODEL:
                output_target, output_target2 = output_target
                output_target = output_target.mean(dim=(2, 3))
                # print(output_target.size(), output_target.size())

            # print("output, target", output_target, target)

            # print('======>', image.size(), target.size(), output_target.size())
            # print(output_target, target)

            # loss = criterion_res(output_target, target)
            loss = criterion_res(output_target, target)

            # print(loss)
            if "LSTM" in cfg.TRAIN.MODEL and False:
                loss += 0.2 * (criterion_res(output_target2[:, 0, :], target)
                               + criterion_res(output_target2[:, 1, :], target)
                               + criterion_res(output_target2[:, 2, :], target)
                               + criterion_res(output_target2[:, 3, :], target)
                               + criterion_res(output_target2[:, 4, :], target))
                # print(loss, criterion_res(output_target2[:, 2, :], target))
            if "MFF" in cfg.TRAIN.MODEL:
                if cfg.DATA.NAME == "Crohn_frame_andSeg":
                    B, H, W = mask.size()
                    # print(B, H, W)
                    # print("output_target2", output_target2.size())
                    pred = F.interpolate(output_target2, (H, W), mode='bilinear', align_corners=True)[:, 0, :, :]
                    # loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
                    loss = F.binary_cross_entropy_with_logits(pred, mask)
                    # print(pred.mean(), mask.mean())
                else:
                    output_target2 = output_target2.mean(dim=(2, 3))
                    loss2 = -(torch.softmax(output_target.detach(), dim=1) * torch.log_softmax(output_target2,
                                                                                               dim=1)).sum(dim=1).mean()
                    # print(loss, loss2)
                    loss = 0.5 * (loss + loss2)

            # top_dice.update(output_target, target)
            # top_iou.update(output_target, target)

            loss = loss / cfg.OPT.GD_STEPS

            loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                scheduler(optimizer, i, epoch, None)  # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

            target_mean = target.type(torch.float).mean()
            output_target = output_target.max(axis=1)[1]
            pred_mean = output_target.type(torch.float).mean()
            target_mean_all.append(target_mean.item())
            pred_mean_all.append(pred_mean.item())

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))

            tbar.set_description("loss: %.3f, learning rate: %.6f" % (
                losses.avg, optimizer.param_groups[-1]['lr']))
            if cfg.DEBUG == False:
                # tensorboard
                tb.add_scalars('Loss_res', {'loss': losses.avg}, epoch)
                # tb.add_scalars('Train_res',
                #                {'top_dice_res': top_dice.avg,
                #                 'top_iou_res': top_iou.avg}, epoch)
                tb.add_scalars('Lr', {'Lr': optimizer.param_groups[-1]['lr']}, epoch)

        _print("loss: %.3f, target: %.3f, pred: %.3f, learning rate: %.6f" % (
            losses.avg, np.mean(target_mean_all), np.mean(pred_mean_all), optimizer.param_groups[-1]['lr']))

        """
        VALIDATION
        """
        if epoch % cfg.VAL.EPOCH == 0:
            top_losses_valid, acc, acc_0, acc_1 = valid_model(_print, cfg, model, criterion, valid_loader)

            # Take dice_score as main_metric to save checkpoint
            # is_best = (top_losses_valid <= best_metric) and (acc >= best_acc)
            # best_metric = min(top_losses_valid, best_metric)
            is_best = (acc_1 >= best_acc1) and (acc >= best_acc)
            best_acc = max(best_acc, acc)
            best_acc1 = max(best_acc1, acc_1)

            # tensorboard
            if cfg.DEBUG == False:
                tb.add_scalars('Valid',
                               {'top_losses_valid': top_losses_valid,
                                'acc': acc}, epoch)

                save_checkpoint({
                    "epoch": epoch + 1,
                    "arch": cfg.EXP,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "optimizer": optimizer.state_dict(),
                }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.TRAIN.FOLD}_epo{epoch}.pth")

    # test_model(_print, cfg, model, test_loader)

    if cfg.DEBUG == False:
        # #export stats to json
        tb.export_scalars_to_json(
            os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}_{cfg.TRAIN.MODEL}_{cfg.COMMENT}_{round(best_metric, 4)}.json"))
        # #close tensorboard
        tb.close()


def test_model_one_getfig(_print, cfg, model, test_loaders, weight=None, tta=False):
    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    if weight is not None:
        print(weight)
        model.load_state_dict(torch.load(weight)["state_dict"])
        model.cuda()

    state = torch.no_grad()
    if cfg.TEST.CAM is False:
        model.eval()
    else:
        model.train()
        state = torch.enable_grad()

    print('load model_cla ...')
    model_cla = torchvision.models.resnet34()
    fc_features = model_cla.fc.in_features
    model_cla.fc = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    model_cla = model_cla.cuda()

    model_dict = '/data/zxk/code/code03/weights/exp022_acc_pre/best_exp022_acc_res_pre_ResNet_fold0.pth'
    ckpt = torch.load(model_dict, "cpu")
    model_cla.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
                              strict=True)
    print('Done')

    test_loader, data_prepare = test_loaders
    tbar = tqdm(test_loader)

    print("test begin")
    os.makedirs(cfg.DIRS.TEST, exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'jpg'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'y'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'feature'), exist_ok=True)
    bingo = miss = 0
    ans = []
    ans_sm = []
    pred_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 12
    pred_matrix = np.array(pred_matrix)
    with state:
        for batch in tbar:
            image, target, frame_name = batch

            image = image.cuda()
            # target = target.cuda()
            output = model(image)
            intense = model_cla(image)
            # print(pred_matrix)
            # print(output, index)
            for i, pred in enumerate(output):
                # print(pred)
                intense_sm = F.softmax(intense[i])
                ans.append(pred.cpu().numpy())
                pred_sm = F.softmax(pred)
                ans_sm.append(pred_sm.cpu().numpy())
                pred_sm = pred_sm.cpu().numpy()
                intense_sm = intense_sm.cpu().numpy()
                # print(pred_sm[0], intense_sm[0])
                p1 = 0 if pred_sm[0] <= 1e-16 else int(pred_sm[0] * 10) + 1
                p2 = 0 if intense_sm[0] <= 1e-16 else int(intense_sm[0] * 10) + 1
                pred_matrix[p1][p2] += 1

                out = batch[0][i].numpy().transpose(1, 2, 0)
                name = frame_name[i].split('/')[-2]
                jpg = frame_name[i].split('/')[-1].split('.')[0]

                root = '/data/zxk/data/Output_health'
                os.makedirs(os.path.join(root, name), exist_ok=True)
                # os.makedirs(os.path.join(root, name, 'health'), exist_ok=True)
                # os.makedirs(os.path.join(root, name, 'disease'), exist_ok=True)
                os.makedirs(os.path.join(root, name, 'crohn_intense'), exist_ok=True)
                os.makedirs(os.path.join(root, name, 'crohn_dirty'), exist_ok=True)
                os.makedirs(os.path.join(root, name, 'health_intense'), exist_ok=True)
                os.makedirs(os.path.join(root, name, 'health_dirty'), exist_ok=True)

                if float(pred_sm[0]) >= 0.9:
                    if float(intense_sm[0]) >= 0.7:
                        cv2.imencode('.jpg', out)[1].tofile(
                            os.path.join(root, name, 'crohn_intense',
                                         jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))
                    elif float(intense_sm[0]) <= 0.01:
                        # print(pred_sm[0], disease_sm[0])
                        cv2.imencode('.jpg', out)[1].tofile(
                            os.path.join(root, name, 'crohn_dirty',
                                         jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))
                # elif p2 == 0:
                #     cv2.imencode('.jpg', out)[1].tofile(os.path.join(root, name, 'disease_possible', jpg + f'_{disease_sm[0]}' + '.jpg'))
                # elif p2 == 11:
                #     cv2.imencode('.jpg', out)[1].tofile(
                #         os.path.join(root, name, 'health_possible', jpg + f'_{disease_sm[0]}' + '.jpg'))
                elif float(pred_sm[0]) <= 1e-7:
                    if float(intense_sm[0]) >= 0.9:
                        cv2.imencode('.jpg', out)[1].tofile(os.path.join(root, name, 'health_intense',
                                                                         jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))
                    elif float(intense_sm[0]) <= 0.01:
                        cv2.imencode('.jpg', out)[1].tofile(os.path.join(root, name, 'health_dirty',
                                                                         jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))

    np_a = np.array(ans_sm)
    np.save(os.path.join(cfg.DIRS.TEST, f"ans.npy"), np.array(ans))
    np.save(os.path.join(cfg.DIRS.TEST, f"ans_sm.npy"), np_a)
    np.savetxt(os.path.join(cfg.DIRS.TEST, f"ans_sm.csv"), np_a, delimiter=',',
               fmt='%.4f')
    print("len =", len(ans))
    print("bingo =", bingo)
    print("miss =", miss)
    print(pred_matrix)


def test_model_one_backup(_print, cfg, model, test_loader, weight=None, tta=False):
    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    if weight is not None:
        # print(weight)
        model.load_state_dict(torch.load(weight)["state_dict"])
        model.cuda()
    Time = []
    state = torch.no_grad()
    if cfg.TEST.CAM is False:
        model.eval()
    else:
        model.train()
        state = torch.enable_grad()

    # print('load model_cla ...')
    # model_cla = torchvision.models.resnet34()
    # fc_features = model_cla.fc.in_features
    # model_cla.fc = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    # model_cla = model_cla.cuda()
    #
    # model_dict = '/data/zxk/code/code03/weights/exp022_acc_pre/best_exp022_acc_res_pre_ResNet_fold0.pth'
    # ckpt = torch.load(model_dict, "cpu")
    # model_cla.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
    #                           strict=True)
    # print('Done')

    tbar = tqdm(test_loader)

    print("test begin")
    os.makedirs(cfg.DIRS.TEST, exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'predicts'), exist_ok=True)
    # os.makedirs(os.path.join(cfg.DIRS.TEST, 'features'), exist_ok=True)
    # bingo = miss = 0
    # ans_all = {}
    ans_sm_all = {}
    ans_done = []
    # pred_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 12
    # pred_matrix = np.array(pred_matrix)
    start_time = time.time()
    with state:
        for batch in tbar:
            image, target, frame_name = batch
            # name = frame_name[0].split('/')[-2]
            # print(image.size(), target[0], name)

            image = image.cuda()
            # target = target.cuda()
            output = model(image)
            # intense = model_cla(image)
            # output = output.detach().cpu().numpy()
            # print(output.shape)
            # np.save(os.path.join('/data/zxk/code/code03/seq/features', f'{target[0]}', f'{name}_0.npy'), output)
            # print(pred_matrix)
            # print(output, index)
            # name = None
            # if name is None:
            #     continue
            for i, pred in enumerate(output):
                # print(pred)
                # intense_sm = F.softmax(intense[i])
                name = frame_name[i].split('/')[-2]
                if name not in ans_sm_all:
                    if cfg.TEST.PRED == 1:
                        for k in ans_sm_all:
                            if k not in ans_done:
                                if len(ans_sm_all[k]) > 0:
                                    # ans_sm_all[k] = sorted(ans_sm_all[k])[:]
                                    # if k not in ans_done:
                                    arr = ans_sm_all[k]
                                    # for ii in range(len(arr)):
                                    #     arr[ii] = (int(arr[ii][1].split('/')[-1][:-4]), arr[ii][1])
                                    # arr = sorted(arr)
                                    with open(os.path.join(cfg.DIRS.TEST, 'samples', f'{k}.txt'), 'w') as out_file:
                                        for a in arr:
                                            out_file.write(str(a[0]) + '+' + str(a[1]) + '\n')
                                    arr = np.array(arr)
                                    np.save(os.path.join(cfg.DIRS.TEST, 'samples', f'{k}.npy'), arr)
                                    ans_done.append(k)
                                    print('output =>', k)
                                    
                    print(name)
                    ans_sm_all[name] = []
                # ans[].append(pred.cpu().numpy())
                t = None
                if cfg.MODEL.FEATURE_EXTRACTOR:
                    pass
                else:
                    pred_sm = F.softmax(pred)
                    # ans_sm_all[name].append(min(9, int(pred_sm.cpu().numpy()[0] * 10)))
                    # pred_sm = pred_sm.cpu().numpy()
                    # intense_sm = intense_sm.cpu().numpy()
                    # print(pred_sm[0], intense_sm[0])
                    # p1 = 0 if pred_sm[0] <= 1e-16 else int(pred_sm[0] * 10) + 1
                    # p2 = 0 if intense_sm[0] <= 1e-16 else int(intense_sm[0] * 10) + 1
                    # pred_matrix[p1][p2] += 1

                    # out = batch[0][i].numpy().transpose(1, 2, 0)
                    # name = frame_name[i].split('/')[-2]
                    # jpg = frame_name[i].split('/')[-1].split('.')[0]

                    # root = '/data/zxk/data/Output_health'
                    # os.makedirs(os.path.join(cfg.DIRS.TEST, 'samples', name), exist_ok=True)
                    # # os.makedirs(os.path.join(root, name, 'health'), exist_ok=True)
                    # # os.makedirs(os.path.join(root, name, 'disease'), exist_ok=True)
                    # os.makedirs(os.path.join(root, name, 'crohn_intense'), exist_ok=True)
                    # os.makedirs(os.path.join(root, name, 'crohn_dirty'), exist_ok=True)
                    # os.makedirs(os.path.join(root, name, 'health_intense'), exist_ok=True)
                    # os.makedirs(os.path.join(root, name, 'health_dirty'), exist_ok=True)
                    #
                    # t = min(9, int(pred_sm.cpu().numpy()[1] * 10))
                    t = float(pred_sm.cpu().numpy()[0])  # 因为之前训好的resnet34的label 0才表示Crohn
                    # t = float(pred_sm.cpu().numpy()[1])
                if t >= 0.0:
                    if cfg.TEST.PRED == 1:
                        ans_sm_all[name].append((t, frame_name[i]))
                    else:
                        ans_sm_all[name].append(t)
                # if t == 0:
                #     cv2.imencode('.jpg', out)[1].tofile(
                #                     os.path.join(cfg.DIRS.TEST, 'jpg', name, jpg + f'_{t}' + '.jpg'))
                # if float(pred_sm[0]) >= 0.9:
                #     if float(intense_sm[0]) >= 0.7:
                #         cv2.imencode('.jpg', out)[1].tofile(
                #             os.path.join(root, name, 'crohn_intense', jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))
                #     elif float(intense_sm[0]) <= 0.01:
                #         # print(pred_sm[0], disease_sm[0])
                #         cv2.imencode('.jpg', out)[1].tofile(
                #             os.path.join(root, name, 'crohn_dirty', jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))
                # # elif p2 == 0:
                # #     cv2.imencode('.jpg', out)[1].tofile(os.path.join(root, name, 'disease_possible', jpg + f'_{disease_sm[0]}' + '.jpg'))
                # # elif p2 == 11:
                # #     cv2.imencode('.jpg', out)[1].tofile(
                # #         os.path.join(root, name, 'health_possible', jpg + f'_{disease_sm[0]}' + '.jpg'))
                # elif float(pred_sm[0]) <= 1e-7:
                #     if float(intense_sm[0]) >= 0.9:
                #         cv2.imencode('.jpg', out)[1].tofile(os.path.join(root, name, 'health_intense', jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))
                #     elif float(intense_sm[0]) <= 0.01:
                #         cv2.imencode('.jpg', out)[1].tofile(os.path.join(root, name, 'health_dirty', jpg + f'_{pred_sm[0]}' + f'_{intense_sm[0]}' + '.jpg'))

    # np_a = np.array(ans_sm)
    # np.save(os.path.join(cfg.DIRS.TEST, f"ans.npy"), np.array(ans))
    # np.save(os.path.join(cfg.DIRS.TEST, f"ans_sm.npy"), np_a)
    # np.savetxt(os.path.join(cfg.DIRS.TEST, f"ans_sm.csv"), np_a, delimiter=',',
    #            fmt='%.4f')
    # print("len =", len(ans))
    # print("bingo =", bingo)
    # print("miss =", miss)
    # print(pred_matrix)

    # ====================================================
    outputs = {}
    for i in ans_sm_all:
        print('name', i)
        if cfg.TEST.PRED == 1:
            name = i
            # arr = sorted(ans_sm_all[i])[:]
            # for i in range(len(arr)):
            #     # print(arr[i][0])
            #     # print(arr[i][1])
            #     # print(arr[i][1].split('/'))
            #     # print(arr[i][1].split('/')[-1])
            #     # print(arr[i][1].split('/')[-1][:-4])
            #     # arr[i][0] = int(arr[i][1].split('/')[-1][:-4])
            #     arr[i] = (int(arr[i][1].split('/')[-1][:-4]), arr[i][1])
            # arr = sorted(arr)
            arr = ans_sm_all[i]
            with open(os.path.join(cfg.DIRS.TEST, 'samples', f'{name}.txt'), 'w') as out_file:
                for a in arr:
                    out_file.write(str(a[0]) + '+' + str(a[1]) + '\n')
            arr = np.array(arr)
            np.save(os.path.join(cfg.DIRS.TEST, 'samples', f'{name}.npy'), arr)
        # =====================================================

        # print(ans_sm_all[i])
        # print(Counter(ans_sm_all[i]))
        # print(dict(Counter(ans_sm_all[i])))
        else:
            temp_dict = dict(Counter(ans_sm_all[i]))
            temp = [0] * 10
            for j in temp_dict:
                temp[j] = temp_dict[j]
            temp_dict['Name'] = i
            temp_dict['all'] = sum(temp)
            temp_dict['p'] = sum(temp[5:]) / sum(temp)
            outputs[i] = temp_dict
    # csv_file = os.path.join(cfg.DIRS.TEST, 'Statistics.csv')

    if cfg.TEST.PRED == 0:
        with open(os.path.join(cfg.DIRS.TEST, f'Statistics_{cfg.EXP}.csv'), 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['Name', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'all', 'p'])
            writer.writeheader()
            for name in outputs:
                value = outputs[name]
                # print(value)
                writer.writerow(value)
    end_time = time.time()
    one_time = end_time - start_time
    print("one_time:", one_time)


def test_model_one(_print, cfg, model, test_loader, weight=None, tta=False):
    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    if weight is not None:
        print(weight)
        model.load_state_dict(torch.load(weight)["state_dict"])
        model.cuda()

    if cfg.MODEL.FEATURE_EXTRACTOR:
        model.fc = DoNothing()
        # model.additional_fc = nn.Identity()

    state = torch.no_grad()
    if cfg.TEST.CAM is False:
        model.eval()
    else:
        model.train()
        state = torch.enable_grad()
    #------------Time------------
    Time = []
    #----------------------------
    # print('load model_cla ...')
    # model_cla = torchvision.models.resnet34()
    # fc_features = model_cla.fc.in_features
    # model_cla.fc = torch.nn.Linear(fc_features, cfg.TRAIN.NUM_CLASS)
    # model_cla = model_cla.cuda()
    #
    # model_dict = '/data/zxk/code/code03/weights/exp022_acc_pre/best_exp022_acc_res_pre_ResNet_fold0.pth'
    # ckpt = torch.load(model_dict, "cpu")
    # model_cla.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
    #                           strict=True)
    # print('Done')
    last_name = ''
    last_feat = []

    tbar = tqdm(test_loader)

    _print("test begin")
    os.makedirs(cfg.DIRS.TEST, exist_ok=True)
    before_name = None
    before_feature = None
    start_time = time.time()
    with state:
        for batch in tbar:
            image, target, frame_name = batch
            name = frame_name[0]
            FOLD = target[0].item()
            # print(image.size(), target[0], name)

            image = image.cuda()
            # target = target.cuda()
            assert target.max() == target.min(), (frame_name, target)
            # assert frame_name.max() == frame_name.min(), (frame_name, target)
            # output = model(image)
            # ------------add-------------------------
            output1 = model(image[:500])
            output2 = model(image[500:])
            # print(output1.shape)
            # print(output2.shape)
            output = torch.cat((output1, output2), 0)
            # print(output.shape)
            # break
            # ------------add-------------------------
            # intense = model_cla(image)
            output = output.squeeze().detach().cpu().numpy()
            # print(output.shape)
            # if name == before_name:
            #     output = np.hstack((output, before_feature))
            #     print('save', output.shape)
            # a = 0 / 0
            if cfg.TEST.PRED <= 0:
                for i in range(image.size(0)):
                    name = frame_name[i]
                    if name != last_name:
                        if last_name != '':
                            print(last_name)
                            output_feat = np.array(last_feat)
                            np.save(os.path.join(cfg.DIRS.OUTPUTS, last_name, f'features.npy'), output_feat)
                            last_feat = []
                    last_name = name
                    last_feat.append(output[i])
            else:
                os.makedirs(os.path.join(cfg.DIRS.OUTPUTS, name), exist_ok=True)
                np.save(os.path.join(cfg.DIRS.OUTPUTS, name,
                                     f'top_{cfg.TEST.BATCH_SIZE}_idx_{FOLD}_of_{cfg.TEST.PRED}.npy'), output)
                _print(f'{name}_top_{cfg.TEST.BATCH_SIZE}_idx_{FOLD}_of_{cfg.TEST.PRED}.npy finished!')
                # np.save(os.path.join(cfg.DIRS.OUTPUTS, f'{name}_0.npy'), output)
                # else:
                #     before_feature = output
                #     before_name = name
    end_time = time.time()
    one_time = end_time - start_time 
    print("one_time: ", one_time)
