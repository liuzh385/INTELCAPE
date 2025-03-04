from tqdm import tqdm
from pytorch_toolbelt.inference import tta as pytta
import os
import time
from .res_gau import *

from .unet3d import unet_3D, UNet3D
from .unet2d import unet_2D
from .bit_ResNet import ResNetV2, ResNetV2_RL
from .LSTM_ResNet import LSTMResNet, TransformerResNet, ResNet_LSTM, ResNet_LSTM_bd, ResNet_TFE, ResNet_TFE_gau, DenseNet_LSTM_bd, \
    DenseNet_TFE
from .RNN_DenseNet import LSTMDenseNet, TransformerDenseNet
from .densenet import *
from tensorboardX import SummaryWriter

from .utils import AverageMeter, DICE, IOU, apply_sigmoid, apply_softmax, save_checkpoint
from .dice_loss_ import *
from .metrics import DiceScoreStorer, IoUStorer
import warnings
import cv2

from .brats_metrics import brats_metics
from .show_cam import show_cam
import matplotlib
import torch.nn.functional as F
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import torchvision
import random
import skvideo.io
from scipy.special import softmax
warnings.filterwarnings("ignore")

import logging
import pandas as pd


class Config(object):
    colors = [[0, 0, 0], [205, 51, 51], [205, 16, 118], [0, 255, 0]]

    labels = [1]


conf = Config()


def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img + 0.5)
    return output_img


def lung_rgb(ct, lung_mask=None, roi_mask=None):
    # print(ct.shape)
    rate = (np.max(ct) - np.min(ct)) / 255
    ct = np.array(ct / rate, np.uint8)
    ct = gamma(ct, 16, 0.5)
    ct_rgb = cv2.cvtColor(ct, cv2.COLOR_GRAY2RGB)

    if lung_mask is not None:
        lung_mask = np.clip(lung_mask, 0, 255)
        lung_mask = np.array(lung_mask, np.uint8)
        # print(np.max(lung_mask), np.min(lung_mask))
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ct_rgb, contours, -1, conf.colors[-1], 3)

    if roi_mask is not None:
        pass
        # print(np.max(roi_mask), np.min(roi_mask))
        # ct_rgb = _merge_mask(ct_rgb, roi_mask)

    return ct_rgb


def save_rgb_png(img_rgb, filename):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img_bgr)


def get_model(cfg):
    print('#######', cfg.TRAIN.MODEL)
    if cfg.TRAIN.MODEL == 'unet_3D':
        model = unet_3D(n_classes=cfg.DATA.SEG_CLASSES, feature_scale=16, in_channels=cfg.DATA.INP_CHANNELS)
    elif cfg.TRAIN.MODEL == 'UNet3D':
        model = UNet3D(cfg.TRAIN.CROP,
                       in_channels=cfg.DATA.INP_CHANNELS,
                       out_channels=cfg.DATA.SEG_CLASSES,
                       init_channels=cfg.MODEL.INIT_CHANNEL,
                       p=cfg.MODEL.DROPOUT)
    elif cfg.TRAIN.MODEL == 'unet_2D':
        model = unet_2D(n_classes=cfg.DATA.SEG_CLASSES, feature_scale=16, in_channels=cfg.DATA.INP_CHANNELS)
    elif cfg.TRAIN.MODEL == 'ResNetV2':
        model = ResNetV2([3, 4, 6, 3], 1, input_size=cfg.DATA.NUM*3)
    elif cfg.TRAIN.MODEL == 'ResNetV2_50x3':
        model = ResNetV2([3, 4, 6, 3], 3, input_size=cfg.DATA.NUM*3)
    elif cfg.TRAIN.MODEL == 'ResNetV2_RL':
        model = ResNetV2_RL([3, 4, 6, 3], 1, input_size=cfg.DATA.NUM*3)
    elif cfg.TRAIN.MODEL == 'ResNet3D':
        model = torchvision.models.video.r3d_18(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, 3)
    elif cfg.TRAIN.MODEL == 'ResNet':
        model = torchvision.models.resnet34(pretrained=False)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, 3)
    elif cfg.TRAIN.MODEL == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=False)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, 3)
    elif cfg.TRAIN.MODEL == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, 3)
    elif cfg.TRAIN.MODEL == 'ResNet_gau':
        model = Resnet_gau(resnet='resnet34')
        ckpt = torch.load("/mnt/minio/node77/liuzheng/RJ/code02_2/weights/best_exp02_100_f0_res_gau_ResNet_gau_fold0.pth", "cpu")
        model.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
            strict=True)
    elif cfg.TRAIN.MODEL == 'ResNet_gau_PT':
        model = Resnet_gau(resnet='resnet34_PT')
    elif cfg.TRAIN.MODEL == 'ResNet18_gau':
        model = Resnet_gau(resnet='resnet18')
    elif cfg.TRAIN.MODEL == 'ResNet50_gau':
        model = Resnet_gau(resnet='resnet50')
    elif cfg.TRAIN.MODEL == 'ResNet_gau_add':
        model = Resnet_gau_add(resnet='resnet34')
    elif cfg.TRAIN.MODEL == 'ResNet18_gau_add':
        model = Resnet_gau_add(resnet='resnet18')
    elif cfg.TRAIN.MODEL == 'VGG11':
        model = torchvision.models.vgg11(pretrained=False)
        fc_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(fc_features, 3)
    elif cfg.TRAIN.MODEL == 'MobileNet':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        fc_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(fc_features, 3)
    elif cfg.TRAIN.MODEL == 'LSTMResNet':
        model = LSTMResNet(3, pretrained=True)
    elif cfg.TRAIN.MODEL == 'LSTMDenseNet':
        model = LSTMDenseNet(3, pretrained=True)
    elif cfg.TRAIN.MODEL == 'ResNet_LSTM':
        model = ResNet_LSTM(cfg, 3, pretrained=False)
    elif cfg.TRAIN.MODEL == 'ResNet_LSTM_bd':
        model = ResNet_LSTM_bd(cfg, 3, pretrained=False)
    elif cfg.TRAIN.MODEL == 'ResNet_TFE':
        model = ResNet_TFE(cfg, 3, pretrained=False)
    elif cfg.TRAIN.MODEL == 'ResNet_TFE_gau':
        model = ResNet_TFE_gau(cfg, 3, pretrained=False)
    elif cfg.TRAIN.MODEL == 'DenseNet_LSTM_bd':
        model = DenseNet_LSTM_bd(cfg, 3, pretrained=False)
    elif cfg.TRAIN.MODEL == 'DenseNet_TFE':
        model = DenseNet_TFE(cfg, 3, pretrained=False)
    elif cfg.TRAIN.MODEL == 'TransformerDenseNet':
        model = TransformerDenseNet(3, pretrained=True)
    elif cfg.TRAIN.MODEL == 'TransformerResNet':
        model = TransformerResNet(3, pretrained=False)
    elif cfg.TRAIN.MODEL == 'DenseNet':
        model = torchvision.models.densenet121(pretrained=True)
        fc_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(fc_features, 3)
    elif cfg.TRAIN.MODEL == 'densenet':
        model = densenet121(pretrained=False)
        fc_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(fc_features, 3)
    else:
        print("Model not found")

    return model


def dice_coeff(pred, target):
    smooth = 1e-6
    num = pred.shape[0]
    m1 = pred.flatten()  # Flatten
    m2 = target.flatten()  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def test_model(_print, cfg, model, test_loaders, weight=None, tta=False):
    # filename = "/mnt/minio/node77/liuzheng/six/pk_img.txt"
    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    if weight is not None:
        model.load_state_dict(torch.load(weight)["state_dict"])

    state = torch.no_grad()
    if cfg.TEST.CAM is False:
        model.eval()
    else:
        model.train()
        state = torch.enable_grad()
    total = 0
    ok = 0 # target = pred
    s_total = 0
    s = 0
    si_total = 0
    si = 0
    li_total = 0
    li = 0
    test_loader, data_prepare = test_loaders
    tbar = tqdm(test_loader)
    ans = [[0, 0, 0]] * 3
    gt_label = [0, 0, 0]
    gt_whole = [0, 0, 0]
    ans = np.array(ans)
    print("test begin")
    os.makedirs(cfg.DIRS.TEST, exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIRS.TEST, 'target'), exist_ok=True)
    bingo = miss = 0
    # frame_num = 10000
    ious = []
    names = []
    bingo_time = 0
    total_time = 0
    start_time = 0
    time_total = 0
    Time_total = 0
    end_time = 0
    data_time = []
    num = 0
    start_Time = 0
    end_Time = 0
    F_num = 0
    F_img = []
    okk = 0
    ttt = 0
    
    start_time_pk = 0
    end_time_pk = 0
    y_true = []
    y_scores = []
    
    y_true_sto = []
    y_scores_sto = []
    
    y_true_small = []
    y_scores_small = []
    
    y_true_large = []
    y_scores_large = []
  
    Class = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with state:
        start_time_pk = time.time()
        for batch in tbar:
            if cfg.DATA.NAME == "Crohn2016_jpg" or cfg.DATA.NAME == "Crohn15to23":
                image, target = batch
                frame_num = 1000
            else:
                avi_name, stomach_time, s_intestine_time, l_intestine_time = batch
                print('==>', avi_name, stomach_time, s_intestine_time, l_intestine_time)
                print(avi_name[0])
                tmp = cv2.VideoCapture(avi_name[0])
                p_name = avi_name[0].split("/")[-2]
                frame_num = int(tmp.get(7))
                print('frame_num:', frame_num)
                # data = str(avi_name)
                # with open(filename, "a", encoding="utf-8") as file:
                #     file.write(data + '\n')
                # data = str(stomach_time) + ' ' + str(s_intestine_time) + ' ' + str(l_intestine_time)
                # with open(filename, "a", encoding="utf-8") as file:
                #     file.write(data + '\n')
            ans_whole = []
            label_whole = []

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
                            start = min(start, frame_num-1)
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
            elif cfg.TEST.PRED == 9:
                tmp = cv2.VideoCapture(avi_name[0])
                # print(avi_name[0], tmp.isOpened())
                frame_num = tmp.get(7)
                frame_data = []
                # int(frame_num) - 20
                num = 20
                for k in tqdm(range(int(frame_num))):
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i = k)
                    image = image.cuda()
                    output = model(image)
                    for i, pred in enumerate(output):
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        pred = softmax(pred)
                        alpha = max(pred[pred_top] - 0.5, 0) + 0.005
                        frame_data.append({'Frame': k,  
                                           'Top Probability Value': pred[1]})
                num += 1
                df = pd.DataFrame(frame_data)

                # 指定 CSV 文件路径
                csv_path = '/mnt/minio/node77/liuzheng/Fig/frame_new.csv'

                # 将 DataFrame 保存到 CSV 文件
                df.to_csv(csv_path, index=False)

                print(f"Data has been written to {csv_path}")
                return 
            elif cfg.TEST.PRED == 2:
                frame_data = []
                start_time = time.time()
                # print("in pred2")
                # print(frame_num)
                start = frame_num // 2
                end = frame_num // 2
                start_a = []
                end_a = []
                n_x1 = 0
                n_x2 = 0

                step = frame_num * 3 // 8
                # print(start, end)
                while step > 1:
                    image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                 i=start)
                    # torch.set_printoptions(profile="full")	
                    # print(image)
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
                        frame_data.append({'Frame': start,  
                                           'Top Probability Value': pred[1]})
                        
                        
                        # print(pred, alpha)
                        # ans[int(pred_top)][int(target[i].item())] += 1
                        # print('start, class:' ,start, target)
                        # print('start, pred_class:' ,start, pred_top)
                        
                        #---------------Sta---------------
                        # if step >= 200:
                        #     if target == 0:
                        #         y_true.append(1)
                        #         y_scores.append(pred[target])
                        #         y_true_sto.append(1)
                        #         y_scores_sto.append(pred[target])
                        #     else:
                        #         y_true.append(0)
                        #         y_scores.append(1 - pred[target])
                        #         y_true_sto.append(0)
                        #         y_scores_sto.append(1 - pred[target])
                                
                        #     if target == 1:
                        #         y_true.append(1)
                        #         y_scores.append(pred[target])
                        #         y_true_small.append(1)
                        #         y_scores_small.append(pred[target])
                        #     else:
                        #         y_true.append(0)
                        #         y_scores.append(1 - pred[target])
                        #         y_true_small.append(0)
                        #         y_scores_small.append(1 - pred[target])
                                
                        #     if target == 2:
                        #         y_true.append(1)
                        #         y_scores.append(pred[target])
                        #         y_true_large.append(1)
                        #         y_scores_large.append(pred[target])
                        #     else:
                        #         y_true.append(0)
                        #         y_scores.append(1 - pred[target])
                        #         y_true_large.append(0)
                        #         y_scores_large.append(1 - pred[target])
                            
                        #     total += 1
                        #     if target == pred_top:
                                
                        #         ok += 1
                        #         if total <= 270:
                        #             okk += 1
                        #     if target == 0:
                        #         s_total += 1
                        #         if target == pred_top:
                        #             s += 1
                        #     if target == 1:
                        #         si_total += 1
                        #         if target == pred_top:
                        #             si += 1
                        #     if target == 2:
                        #         li_total += 1
                        #         if target == pred_top:
                        #             li += 1
                        
                            # if total == 270:
                            #     print("270 acc:", okk / 270)
                            #     end_time_pk = time.time()
                            #     print("270 time:", end_time_pk - start_time_pk)
                            #     break
                        #---------------Sta---------------
                        if pred_top == 0:
                            # print('++++++++++++++++++++++++++++++', start, int(step * 2 * alpha))
                            start += int(step * 2 * alpha)
                            # print('==============================', start, int(step * 2 * alpha))
                            start = min(start, frame_num - step // 10)
                        else:
                            start -= int(step * 2 * alpha)
                            start = max(1 + step // 10, start)

                        step = step * 9 // 10
                    # print('step, start, end', step, start, end)

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
                        num += 1
                        pred = pred.clone().cpu().numpy()
                        pred_top = np.argmax(pred)
                        pred = softmax(pred)
                        alpha = max(pred[pred_top] - 0.5, 0) + 0.005
                        frame_data.append({'Frame': end,  
                                           'Top Probability Value': pred[1]})
                        # print(pred, alpha)
                        # print('end, class:' ,end, target)
                        # print('end, pred_class:' ,end, pred_top)
                        
                        #---------------Sta---------------
                        # if step >= 200:
                        #     if target == 0:
                        #         y_true.append(1)
                        #         y_scores.append(pred[target])
                        #         y_true_sto.append(1)
                        #         y_scores_sto.append(pred[target])
                        #     else:
                        #         y_true.append(0)
                        #         y_scores.append(1 - pred[target])
                        #         y_true_sto.append(0)
                        #         y_scores_sto.append(1 - pred[target])
                            
                        #     if target == 1:
                        #         y_true.append(1)
                        #         y_scores.append(pred[target])
                        #         y_true_small.append(1)
                        #         y_scores_small.append(pred[target])
                        #     else:
                        #         y_true.append(0)
                        #         y_scores.append(1 - pred[target])
                        #         y_true_small.append(0)
                        #         y_scores_small.append(1 - pred[target])
                            
                        #     if target == 2:
                        #         y_true.append(1)
                        #         y_scores.append(pred[target])
                        #         y_true_large.append(1)
                        #         y_scores_large.append(pred[target])
                        #     else:
                        #         y_true.append(0)
                        #         y_scores.append(1 - pred[target])
                        #         y_true_large.append(0)
                        #         y_scores_large.append(1 - pred[target])
                                
                            
                        #     total += 1
                        #     if target == pred_top:
                        #         ok += 1
                        #         if total <= 270:
                        #             okk += 1
                        #     if target == 0:
                        #         s_total += 1
                        #         if target == pred_top:
                        #             s += 1
                        #     if target == 1:
                        #         si_total += 1
                        #         if target == pred_top:
                        #             si += 1
                        #     if target == 2:
                        #         li_total += 1
                        #         if target == pred_top:
                        #             li += 1
                       
                        
                            # if total == 270:
                            #     end_time_pk = time.time()
                            #     print("270 acc:", okk / 270)
                            #     print("270 time:", end_time_pk - start_time_pk)
                            #     break
                        #---------------Sta---------------
                        
                        end_before = start
                        if pred_top == 2:
                            end -= int(step * 2 * alpha)
                            end = max(1 + step // 10, end)
                        else:
                            end += int(step * 2 * alpha)
                            end = min(end, frame_num - step // 10)

                        step = step * 9 // 10
                    # print('step, start, end', step, start, end)

                print('===>', start, end, s_intestine_time, l_intestine_time)
                end_time = time.time()
                one_time = (end_time - start_time) 
                data_time.append(one_time)
                # print('3 class acc',ok / total)
                # ok = 0
                # total = 0
                # if s_total > 0:
                #     print('stomach acc:', s / s_total)
                #     s = 0
                #     s_total = 0
                # if si_total > 0:
                #     print('s_intestine acc:', si / si_total)
                #     si = 0
                #     si_total = 0
                # if li_total > 0:
                #     print('l_intestine acc:', li / li_total)
                #     li = 0
                #     li_total = 0
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
                df = pd.DataFrame(frame_data)

                # 指定 CSV 文件路径
                

                # 将 DataFrame 保存到 CSV 文件
                

                
                plt.figure()
                plt.plot(list(range(n_x1)), start_a, label="start")
                plt.plot(list(range(n_x2)), end_a, label="end")
                plt.plot(list(range(n_x1)), [s_intestine_time.item()] * n_x1, label="start_gt", alpha=0.3)
                plt.plot(list(range(n_x2)), [l_intestine_time.item()] * n_x2, label="end_gt", alpha=0.3)
                plt.xlabel("iter")
                plt.ylabel("num_frame")
                plt.legend()
                plt.savefig(os.path.join(save_root, save_name, f"01.jpg"))
                
                
            elif cfg.TEST.PRED == 0:
                for index in range(frame_num//1000):
                    # if index % 100 == 0:
                        # print(index*1000, frame_num)
                    if cfg.DATA.NAME != "Crohn2016_jpg" and cfg.DATA.NAME != "Crohn15to23":
                        image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time,
                                                     i=index * 10000)
                        # image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time)
                    # print(image.size())
                    # print(image[:, :, :, 160, 160])
                    image = image.cuda()
                    # target = target.cuda()
                    output = model(image)
                    # print("output, target", output, target)
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

            elif cfg.TEST.PRED == 4:
                videodata = skvideo.io.vread(avi_name[0])
                videodata = videodata.transpose(0, 3, 1, 2)
                b_len = 200
                print("videodata", videodata.shape)
                videodata = videodata[:, [2, 1, 0], :, :]
                ans = []
                ans_sm = []
                for index in range(0, frame_num, b_len):
                    if index % 100 == 0:
                        print(index, '/', frame_num)
                    # frame1 = None
                    # for i in range(b_len):
                    #     image, _ = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time, i=index+i)
                    #     if i == 0:
                    #         frame1 = image
                    #     else:
                    #         frame1 = np.concatenate((frame1, image), axis=0)

                    frame = videodata[index: min(index+b_len, frame_num)]
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

    if cfg.TEST.PRED != 0:
        end_Time = time.time()
        stack_ious = torch.stack(ious)
        for i in range(len(ious)):
            print(names[i])
            print(ious[i])
        print('mean, std =>', stack_ious.mean(), stack_ious.std())
        print('3 class acc',ok / total)
        print('stomach acc:', s / s_total)
        print('s_intestine acc:', si / si_total)
        print('l_intestine acc:', li / li_total)
        print('270 acc:', okk / 270)
        print('Time: ',end_Time - start_Time)
        
        # df_all = pd.DataFrame({
        #             'y_true' : y_true,
        #             'y_scores' : y_scores
        #         })
        # df_all.to_csv('/mnt/minio/node77/liuzheng/ROC/all/rj.csv', index=False)
                
        # df_sto = pd.DataFrame({
        #             'y_true' : y_true_sto,
        #             'y_scores' : y_scores_sto
        #         })
        # df_sto.to_csv('/mnt/minio/node77/liuzheng/ROC/Stomach/rj.csv', index=False)
                
        # df_small = pd.DataFrame({
        #             'y_true' : y_true_small,
        #             'y_scores' : y_scores_small
        #         })
        # df_small.to_csv('/mnt/minio/node77/liuzheng/ROC/Small_intestine/rj.csv', index=False)
                
        # df_large = pd.DataFrame({
        #             'y_true' : y_true_large,
        #             'y_scores' : y_scores_large
        #         })
        # df_large.to_csv('/mnt/minio/node77/liuzheng/ROC/Large_intestine/rj.csv', index=False)
        
        mean = np.mean(data_time)
        std = np.std(data_time)
        print(f"平均数：{mean}")
        print(f"标准差：{std}")
        # plt.plot(range(len(ious)), ious)
        # plt.savefig(f"./test_whole.png")
        np.savetxt(os.path.join(cfg.DIRS.TEST, f'ans_{stack_ious.mean()}_{stack_ious.std()}.txt'), ious)
    else:
        acc = bingo / (bingo + miss)
        print(ans, acc)
        print(gt_label)
        print(gt_whole)
        # print([a / b for a, b in zip(gt_label, gt_whole)])
        # print(np.mean([a / b for a, b in zip(gt_label, gt_whole)]))
        np.savetxt(os.path.join(cfg.DIRS.TEST, f'ans_{acc}.txt'), ans)


def valid_model(_print, cfg, model, valid_criterion, valid_loaders, tta=False):
    losses = AverageMeter()
    top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
    top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    model.eval()
    if cfg.DATA.NPY is True:
        valid_loader, _ = valid_loaders
    else:
        valid_loader, data_prepare = valid_loaders
    tbar = tqdm(valid_loader)

    target_mean_all = []
    pred_mean_all = []

    bingo = 0
    miss = 0

    with torch.no_grad():
        for i, batch in enumerate(tbar):
            if cfg.DATA.NPY is True:
                image, target = batch
            elif cfg.DATA.NAME == "Crohn2016_jpg" or cfg.DATA.NAME == "Crohn15to23":
                image, target = batch
            else:
                avi_name, stomach_time, s_intestine_time, l_intestine_time = batch
                image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time)
            # print(image.size(), target.size())
            # print(_id, finding)
            # print(image.max(), image.min(), image.mean(), target.max(), target.min(), target.mean())
            image = image.cuda()
            target = target.cuda()
            # output = model(image)
            output = model(image)
            # print(output, target)
            # if "LSTM" in cfg.TRAIN.MODEL:
            #     output, _ = output

            # loss
            loss = valid_criterion(output, target)

            for i, pred in enumerate(output):
                pred = pred.clone().cpu().numpy()
                pred_top = np.argmax(pred)
                if pred_top == target[i].item():
                    bingo += 1
                else:
                    miss += 1

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

    acc = bingo / (bingo + miss)
    # _print("Valid iou: %.3f, dice: %.3f loss: %.3f" % (top_iou.avg, top_dice.avg, losses.avg))
    _print("Target mean: %.3f, Pred mean: %.3f, acc: %.3f loss: %.3f" % (
            np.mean(target_mean_all), np.mean(pred_mean_all), acc, losses.avg))

    return losses.avg, acc


def train_loop(_print, cfg, model, train_loaders, valid_loader, criterion, valid_criterion, optimizer, scheduler,
               start_epoch, best_metric, test_loaders):
    # print("in train")
    if cfg.DEBUG == False:
        tb = SummaryWriter(f"runs/{cfg.EXP}/{cfg.TRAIN.MODEL}", comment=f"{cfg.COMMENT}")  # for visualization
    # weight = cfg.MODEL.WEIGHT
    # model.load_state_dict(torch.load(weight)["state_dict"], strict=False)
    best_acc = 0
    print(start_epoch, cfg.TRAIN.EPOCHS)
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # print("in train")
        _print(f"Epoch {epoch + 1}")

        # define some meters
        losses = AverageMeter()

        # top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
        # top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)

        """
        TEST
        """
        if cfg.TEST.TIMES and epoch !=0 and epoch % (cfg.TRAIN.EPOCHS//cfg.TEST.TIMES) == 0:
            cfg_epoch = cfg.clone()
            cfg_epoch.defrost()
            cfg_epoch.DIRS.TEST = os.path.join(cfg_epoch.DIRS.TEST, "test_epoch" + str(epoch))
            test_model(_print, cfg_epoch, model, test_loaders, weight=None, tta=cfg.INFER.TTA)
            # cfg_epoch = cfg.clone()
            # cfg_epoch.defrost()
            # cfg_epoch.DIRS.TEST = os.path.join(cfg_epoch.DIRS.TEST, "train_epoch" + str(epoch))
            # test_model(_print, cfg_epoch, model, train_loaders, weight=None, tta=cfg.INFER.TTA)

        """
        TRAINING
        """
        # switch model to training mode
        model.train()

        if cfg.DATA.NPY is True:
            train_loader, _ = train_loaders
        else:
            train_loader, data_prepare = train_loaders
        tbar = tqdm(train_loader)

        for i, batch in enumerate(tbar):
            # print("num:", i)
            criterion_res = criterion
            if cfg.DATA.NPY is True:
                image, target = batch
            elif cfg.DATA.NAME == "Crohn2016_jpg" or cfg.DATA.NAME == "Crohn15to23":
                image, target = batch
                # print(target)
            else:
                avi_name, stomach_time, s_intestine_time, l_intestine_time = batch
                print("begin get batch:")
                image, target = data_prepare(avi_name, stomach_time, s_intestine_time, l_intestine_time)
                print("finish get batch")

            # print(image.size(), target.size())
            # print('image[100, 100]', image[0, :, 100, 100])

            image = image.cuda()
            target = target.cuda()

            output_target = model(image)
            if "LSTM" in cfg.TRAIN.MODEL and False:
                output_target, output_target2 = output_target

            # print("output, target", output_target, target)

            # print('======>', image.size(), target.size(), output_target.size())
            # print(target, output_target)

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

            # top_dice.update(output_target, target)
            # top_iou.update(output_target, target)

            loss = loss / cfg.OPT.GD_STEPS

            loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                scheduler(optimizer, i, epoch, None)  # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

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

            if (i + 1) % cfg.VAL.ITEM == 0:
                _print("loss: %.3f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))
                """
                        VALIDATION
                        """
                top_losses_valid, acc = valid_model(_print, cfg, model, criterion, valid_loader)

                # Take dice_score as main_metric to save checkpoint
                is_best = top_losses_valid < best_metric
                best_metric = min(top_losses_valid, best_metric)

                # tensorboard
                if cfg.DEBUG == False:
                    tb.add_scalars('Valid',
                                   {'top_losses_valid': top_losses_valid,
                                    'top_acc': acc}, epoch)

                    save_checkpoint({
                        "epoch": epoch + 1,
                        "arch": cfg.EXP,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "optimizer": optimizer.state_dict(),
                    }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.TRAIN.FOLD}.pth")


        _print("loss: %.3f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

        """
        VALIDATION
        """
        top_losses_valid, acc = valid_model(_print, cfg, model, criterion, valid_loader)

        # Take dice_score as main_metric to save checkpoint
        is_best = (top_losses_valid <= best_metric) and (acc >= best_acc)
        best_metric = min(top_losses_valid, best_metric)
        best_acc = max(best_acc, acc)

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
            }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.TRAIN.FOLD}.pth")

    # test_model(_print, cfg, model, test_loader)

    if cfg.DEBUG == False:
        # #export stats to json
        tb.export_scalars_to_json(
            os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}_{cfg.TRAIN.MODEL}_{cfg.COMMENT}_{round(best_metric, 4)}.json"))
        # #close tensorboard
        tb.close()


def valid_RL_model(_print, cfg, model, valid_criterion, valid_loaders, tta=False):
    losses = AverageMeter()
    ious = AverageMeter()

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    model.eval()
    valid_loader, data_prepare = valid_loaders
    tbar = tqdm(valid_loader)

    target_mean_all = []
    pred_mean_all = []

    with torch.no_grad():
        for i, batch in enumerate(tbar):
            avi_name, stomach_time, s_intestine_frame, l_intestine_frame = batch
            criterion_res = valid_criterion

            tmp = cv2.VideoCapture(avi_name[0])
            frame_num = tmp.get(7)
            batch_size = cfg.TRAIN.BATCH_SIZE

            i_start = 0.1
            i_end = 0.5
            iou = 0

            # network forward
            loss_sum = 0
            for step in range(cfg.TRAIN.NUM_STEPS):

                image_start, target_start = data_prepare(avi_name, stomach_time, s_intestine_frame, l_intestine_frame,
                                                         i=int(frame_num * i_start))
                image_end, target_end = data_prepare(avi_name, stomach_time, s_intestine_frame, l_intestine_frame,
                                                     i=int(frame_num * i_end))

                image_start = image_start.cuda()
                target_start = target_start.cuda()
                image_end = image_end.cuda()
                target_end = target_end.cuda()

                output_start, value_start = model(image_start)
                output_end, value_end = model(image_end)

                for i, pred in enumerate(output_start):
                    pred = pred.clone().cpu().detach().numpy()
                    pred_top = np.argmax(pred)

                    if pred_top == 0:
                        i_start += (1 - i_start) * value_start[i]
                    else:
                        i_start -= (i_start) * value_start[i]

                for i, pred in enumerate(output_end):
                    pred = pred.clone().cpu().detach().numpy()
                    pred_top = np.argmax(pred)

                    if pred_top == 2:
                        i_end -= (i_end) * value_end[i]
                    else:
                        i_end += (1 - i_end) * value_end[i]

                tmp = [int(frame_num * i_start), int(frame_num * i_end), s_intestine_frame, l_intestine_frame]
                tmp = np.array(tmp)
                tmp = np.sort(tmp)
                iou = (tmp[2] - tmp[1]) / (tmp[3] - tmp[0])

                loss = criterion_res(output_start, target_start) + criterion_res(output_end, target_end)
                loss += 2 * (1 - torch.tensor(iou).cuda())
                loss /= 40
                loss_sum += loss.item()

            # record
            losses.update(loss_sum, image_start.size(0))
            ious.update(iou, image_start.size(0))

    # _print("Valid iou: %.3f, dice: %.3f loss: %.3f" % (top_iou.avg, top_dice.avg, losses.avg))
    _print("ious mean: %.3f, loss: %.3f" % (ious.avg, losses.avg))

    return ious.avg


# Training RL
def train_RL_loop(_print, cfg, model, train_loaders, valid_loader, criterion, valid_criterion, optimizer, scheduler,
               start_epoch, best_metric, test_loaders):
    if cfg.DEBUG == False:
        tb = SummaryWriter(f"runs/{cfg.EXP}/{cfg.TRAIN.MODEL}", comment=f"{cfg.COMMENT}")  # for visualization

    model.train()

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"Epoch {epoch + 1}")

        losses = AverageMeter()
        ious = AverageMeter()
        train_loader, data_prepare = train_loaders
        tbar = tqdm(train_loader)

        for i, batch in enumerate(tbar):
            avi_name, stomach_time, s_intestine_frame, l_intestine_frame = batch
            criterion_res = criterion

            tmp = cv2.VideoCapture(avi_name[0])
            batch_size = cfg.TRAIN.BATCH_SIZE
            frame_num = int(tmp.get(7))

            i_start = 0.1
            i_end = 0.5

            # network forward
            loss = 0
            for step in range(cfg.TRAIN.NUM_STEPS):

                # print(frame_num, i_start)
                image_start, target_start = data_prepare(avi_name, stomach_time, s_intestine_frame, l_intestine_frame,
                                                         i=frame_num * i_start)
                image_end, target_end = data_prepare(avi_name, stomach_time, s_intestine_frame, l_intestine_frame,
                                                     i=frame_num * i_end)

                image_start = image_start.cuda()
                target_start = target_start.cuda()
                image_end = image_end.cuda()
                target_end = target_end.cuda()

                output_start, value_start = model(image_start)
                output_end, value_end = model(image_end)

                for i, pred in enumerate(output_start):
                    pred = pred.clone().cpu().detach().numpy()
                    pred_top = np.argmax(pred)

                    if pred_top == 0:
                        i_start += (1 - i_start) * value_start[i]
                    else:
                        i_start -= (i_start) * value_start[i]

                for i, pred in enumerate(output_end):
                    pred = pred.clone().cpu().detach().numpy()
                    pred_top = np.argmax(pred)

                    if pred_top == 2:
                        i_end -= (i_end) * value_end[i]
                    else:
                        i_end += (1 - i_end) * value_end[i]

                # tmp = [int(frame_num * i_start), int(frame_num * i_end), s_intestine_frame, l_intestine_frame]
                # tmp = np.array(tmp)
                # tmp = np.sort(tmp)
                # iou = (tmp[2] - tmp[1]) / (tmp[3] - tmp[0])

                # print(output_start, target_start)
                loss_one = criterion_res(output_start, target_start) + criterion_res(output_end, target_end)
                # loss_one += 2 * (1 - torch.tensor(iou).cuda())
                if step != cfg.TRAIN.NUM_STEPS - 1:
                    loss_one /= 10
                    loss_one = loss_one / cfg.OPT.GD_STEPS
                    loss_one.backward()
                loss = loss_one

            tmp = [int(frame_num * i_start), int(frame_num * i_end), s_intestine_frame, l_intestine_frame]
            tmp = np.array(tmp)
            tmp = np.sort(tmp)
            iou = (tmp[2] - tmp[1]) / (tmp[3] - tmp[0])

            loss += 2 * (1 - torch.tensor(iou).cuda())
            # loss_one /= 40
            loss = loss / cfg.OPT.GD_STEPS

            loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                scheduler(optimizer, i, epoch, None)  # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image_start.size(0))
            ious.update(iou, image_start.size(0))

            tbar.set_description("Train iou: %.3f, loss: %.3f, learning rate: %.6f" % (
                ious.avg, losses.avg, optimizer.param_groups[-1]['lr']))
            if cfg.DEBUG == False:
                # tensorboard
                tb.add_scalars('Loss_res', {'loss': losses.avg}, epoch)
                tb.add_scalars('Train_res',
                               {'top_iou_res': ious.avg}, epoch)
                tb.add_scalars('Lr', {'Lr': optimizer.param_groups[-1]['lr']}, epoch)

        _print("Train iou: %.3f, loss: %.3f, learning rate: %.6f" % (
            ious.avg, losses.avg, optimizer.param_groups[-1]['lr']))

        """
        VALIDATION
        """
        top_iou_valid = valid_RL_model(_print, cfg, model, criterion, valid_loader)

        # Take dice_score as main_metric to save checkpoint
        is_best = top_iou_valid > best_metric
        best_metric = max(top_iou_valid, best_metric)

        # tensorboard
        if cfg.DEBUG == False:
            tb.add_scalars('Valid',
                           {'top_iou': top_iou_valid}, epoch)

            save_checkpoint({
                "epoch": epoch + 1,
                "arch": cfg.EXP,
                "state_dict": model.state_dict(),
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
            }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.TRAIN.FOLD}.pth")

        # test_model(_print, cfg, model, test_loader)

        if cfg.DEBUG == False:
            # #export stats to json
            tb.export_scalars_to_json(
                os.path.join(cfg.DIRS.OUTPUTS,
                             f"{cfg.EXP}_{cfg.TRAIN.MODEL}_{cfg.COMMENT}_{round(best_metric, 4)}.json"))
            # #close tensorboard
            tb.close()
