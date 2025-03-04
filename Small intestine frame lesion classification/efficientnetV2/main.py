import sys
import os
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn

from helpers import setup_determinism
from train_and_test import train, test
from efficientnet_pytorch import EfficientNet
from model import efficientnetv2_s as create_model

# train
# CUDA_VISIBLE_DEVICES=4 python main.py --logname t2tvit.log --tspath ./runs/exp02 --testlogdir ./test/t2tvit --weightname t2tvit.pth --img_size 224
# CUDA_VISIBLE_DEVICES=2 python main.py --logname resnet50.log --tspath ./runs/exp03 --testlogdir ./test/resnet50 --weightname resnet50.pth --img_size 224
# CUDA_VISIBLE_DEVICES=2 python main.py --logname resnet34_pretrain.log --tspath ./runs/exp07 --testlogdir ./test/resnet34_pretrain --weightname resnet34_pretrain.pth --img_size 224
# CUDA_VISIBLE_DEVICES=7 python main.py --logname resnet18.log --tspath ./runs/exp05 --testlogdir ./test/resnet18 --weightname resnet18.pth --img_size 224
# CUDA_VISIBLE_DEVICES=2 python main.py --logname resnet34_colorjitter.log --tspath ./runs/exp06 --testlogdir ./test/resnet34_colorjitter --weightname resnet34_colorjitter.pth --img_size 224
# test
# CUDA_VISIBLE_DEVICES=7 python main.py --logname bestvit.log --testlogdir ./test/bestvit --load /mnt/minio/node77/caiyinqi/vit_train/weights/best_vitb16.pth --img_size 256 --test
# CUDA_VISIBLE_DEVICES=7 python main.py --logname bestres50.log --testlogdir ./test/bestres50 --load /mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet50.pth --img_size 224 --test
# CUDA_VISIBLE_DEVICES=7 python main.py --logname bestres34.log --testlogdir ./test/bestres34 --load /mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet34.pth --img_size 224 --test
# CUDA_VISIBLE_DEVICES=1 python main.py --logname bestres34_pretrain.log --testlogdir ./test/bestres34_pretrain --load /mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet34_pretrain.pth --img_size 224 --test
# CUDA_VISIBLE_DEVICES=2 python main.py --logname bestres34_colorjitter.log --testlogdir ./test/bestres34_colorjitter --load /mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet34_colorjitter.pth --img_size 224 --test
# CUDA_VISIBLE_DEVICES=4 python main.py --logname bestres18.log --testlogdir ./test/bestres18 --load /mnt/minio/node77/caiyinqi/vit_train/weights/best_resnet18.pth --img_size 224 --test
# CUDA_VISIBLE_DEVICES=7 python main.py --logname bestt2tvit.log --testlogdir ./test/bestt2tvit --load /mnt/minio/node77/caiyinqi/vit_train/weights/best_t2tvit.pth --img_size 224 --test
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logs/exp02",
                        help="path to log")
    parser.add_argument("--logname", type=str, default="effb4-rj",
                        help="filename to log")
    parser.add_argument("--tspath", type=str, default="./runs/exp02",
                        help="path to tensorborad log")
    parser.add_argument("--testlogdir", type=str, default="./test/exp02",
                        help="path to test log")
    parser.add_argument("--savedir", type=str, default="./weights/exp02",
                        help="dir to save checkpoint")
    parser.add_argument("--weightname", type=str, default="RJ_efficientnetb4.pth",
                        help="path to save checkpoint")
    parser.add_argument("--load", type=str, default="",
                        help="path to model weight")
    parser.add_argument("--seed", type=int, default=721,
                        help="seed")
    parser.add_argument("--img_size", type=int, default=224, # before 256
                        help="image size")                  
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="model runing mode (train/test)")
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")

    args = parser.parse_args()
    if args.test:
        args.mode = "test"

    return args


def setup_logging(args):
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(
        os.path.join(args.logdir, args.logname),
        mode='a'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info(f'===============================')
    logging.info(f'Command arguments {args}')


def main(args):
    # 设置超参
    lr = 1e-3
    beta1 = 0.95
    beta2 = 0.98
    weight_decay=3e-2
    batch_size = 32
    args.num_epochs = 100
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    # 加载数据集
    train_transform = transforms.Compose([
        # transforms.CenterCrop(560), #图片分辨率为360*360，裁剪掉4边的20像素，剩下320*320
        transforms.Resize(img_size[num_model][0]),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        # transforms.CenterCrop(560), #图片分辨率为360*360，裁剪掉4边的20像素，剩下320*320
        transforms.Resize(img_size[num_model][1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Sysu 6
    # if args.mode == 'train':
    #     train_dataset = datasets.ImageFolder(root='/mnt/minio/node77/liuzheng/Crohn2Class/split_cls_frames_sysu/train', transform=train_transform)
    #     valid_dataset = datasets.ImageFolder(root='/mnt/minio/node77/liuzheng/Crohn2Class/split_cls_frames_sysu/valid', transform=test_transform)
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # test_dataset = datasets.ImageFolder(root='/mnt/minio/node77/liuzheng/Crohn2Class/split_cls_frames_sysu/test', transform=test_transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # RJ
    if args.mode == 'train':
        train_dataset = datasets.ImageFolder(root='/mnt/minio/node77/liuzheng/Crohn2Class/split_cls_frames_RJ/train', transform=train_transform)
        valid_dataset = datasets.ImageFolder(root='/mnt/minio/node77/liuzheng/Crohn2Class/split_cls_frames_RJ/valid', transform=test_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataset = datasets.ImageFolder(root='/mnt/minio/node77/liuzheng/Crohn2Class/split_cls_frames_RJ/test', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 模型
    model = create_model(num_classes=2).cuda()
    print(model)

    # 修改最后的全连接层以适应二分类
    # feature = model._fc.in_features
    # model._fc = nn.Linear(in_features=feature,out_features=2,bias=True)
    
    # # 冻结除分类层外的所有层
    # for param in model.parameters():
    #     param.requires_grad = False

    # # 只解冻分类层
    # for param in model._fc.parameters():
    #     param.requires_grad = True
    # 加载预训练
    # pretrained_weights = torch.load("/mnt/minio/node77/liuzheng/Crohn2Class/weights/exp01/best_efficientnet_sysu6efficientnetb2.pth")
    # model.load_state_dict(pretrained_weights)
    model = model.cuda()
    
    if args.load != "":
        if os.path.isfile(args.load):
            logging.info(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
                                  strict=True)
            logging.info(f"=> load model from checkpoint: '{args.load}'")

    # 设置优化器
    # optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    lr = 1e-2 # 学习率
    momentum = 0.9  # 动量
    weight_decay = 1e-4  # 权重衰减 3e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)

    # 设置损失函数
    loss = nn.CrossEntropyLoss()

    if args.mode == 'train':
        train(logging.info, model, train_dataloader, valid_dataloader, loss, optimizer, scheduler, test_dataloader, args)
        test(logging.info, model, test_dataloader, loss, args)
    else:
        test(logging.info, model, test_dataloader, loss, args)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args)
    setup_determinism(args.seed)
    main(args)