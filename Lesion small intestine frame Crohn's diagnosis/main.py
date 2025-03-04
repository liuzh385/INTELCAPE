import os
import sys
import argparse
import logging
import torch.optim as optim
import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
# from apex import amp
import torch
from config import get_cfg_defaults
from models import train_loop, valid_model, test_model_one, test_model_one_backup, test_model, get_model, DoNothing
from datasets import get_dataset
from lr_scheduler import LR_Scheduler
from helpers import setup_determinism
from losses import get_loss
import warnings
# from torchvision.models.feature_extraction import create_feature_extractor
warnings.filterwarnings("ignore")

env_dist = os.environ
env_dist.get('config')
env_dist.get('test')

"""
    Code07 is used to classify the Crohn's Disease from CE
    
    Input: CE Videos (frames)
    Output: Crohn / Non-Crohn of each patient
            Lesion / Health of each frame
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--load", type=str, default="",
                        help="path to model weight")
    parser.add_argument("-ft", "--finetune", action="store_true",
                        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="model runing mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
                        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args


def setup_logging(args, cfg):
    if not os.path.isdir(cfg.DIRS.LOGS):
        os.mkdir(cfg.DIRS.LOGS)

    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(
        os.path.join(cfg.DIRS.LOGS, f'{cfg.EXP}_{cfg.TRAIN.MODEL}_{args.mode}_fold{cfg.TRAIN.FOLD}.log'),
        mode='a'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info(f'===============================')
    logging.info(f'\n\nStart with config {cfg}')
    logging.info(f'Command arguments {args}')


def main(args, cfg):
    logging.info(f"=========> {cfg.EXP} <=========")

    # Declare variables
    start_epoch = 0
    best_metric = 1.

    # Create model
    model = get_model(cfg)
    train_criterion = get_loss(cfg)
    valid_criterion = get_loss(cfg)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()
        valid_criterion = valid_criterion.cuda()

    # #optimizer
    if cfg.OPT.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(params=model.parameters(),
                                lr=cfg.OPT.BASE_LR,
                                weight_decay=cfg.OPT.WEIGHT_DECAY)
    elif cfg.OPT.OPTIMIZER == "adam":
        optimizer = optim.Adam(params=model.parameters(),
                               lr=cfg.OPT.BASE_LR,
                               weight_decay=cfg.OPT.WEIGHT_DECAY)
    elif cfg.OPT.OPTIMIZER == "adagrad":
        optimizer = optim.Adagrad(params=model.parameters(),
                                  lr=cfg.OPT.BASE_LR,
                                  weight_decay=cfg.OPT.WEIGHT_DECAY)
    else:
        raise Exception("OPT.OPTIMIZER ERROR")

    # if cfg.SYSTEM.FP16:
    #     model, optimizer = amp.initialize(models=model, optimizers=optimizer,
    #                                       opt_level=cfg.SYSTEM.OPT_L,
    #                                       keep_batchnorm_fp32=(True if cfg.SYSTEM.OPT_L == "O2" else None))

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            print(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()}, strict=True)

            if not args.finetune:
                print("resuming optimizer ...")
                optimizer.load_state_dict(ckpt.pop('optimizer'))
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
                logging.info(
                    f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    # Load data
    test_loaders = get_dataset('test', cfg)
    num = 0
    # for batch in test_loaders:
    #     image, target, frame_name = batch
    #     print(image.shape)
    #     print(target.shape)
    #     print(frame_name)
    #     break
    if args.mode != "test":
        train_loaders = get_dataset('train', cfg)
        valid_loaders = get_dataset('valid', cfg)
        scheduler = LR_Scheduler("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS, iters_per_epoch=len(train_loaders),
                                 warmup_epochs=cfg.OPT.WARMUP_EPOCHS)

        if args.mode == "train":
            train_loop(logging.info, cfg, model, train_loaders, valid_loaders, train_criterion, valid_criterion,
                       optimizer, scheduler, start_epoch, best_metric, test_loaders)
            test_model(logging.info, cfg, model, test_loaders, weight=None)
        elif args.mode == "valid":
            valid_model(logging.info, cfg, model, valid_criterion, valid_loaders)
    else:
        # 不同的任务需要修改test_model

        # print(model)
        # test_model_one(logging.info, cfg, model, test_loaders, weight=cfg.MODEL.WEIGHT)  # for test02_gettop2000(select top2000frames)
        test_model(logging.info, cfg, model, test_loaders, weight=cfg.MODEL.WEIGHT)  # for testexp01 and testexp02(classify crohn) and test03
        # test_model_one_backup(logging.info, cfg, model, test_loaders, weight=cfg.MODEL.WEIGHT)  # for test01_getframesnpy
        
        # cfg_epoch = cfg.clone()
        # cfg_epoch.defrost()
        # cfg_epoch.DIRS.TEST = os.path.join(cfg_epoch.DIRS.TEST, "train_epoch")
        # test_model(logging.info, cfg_epoch, model, train_loader, weight=None, tta=cfg.INFER.TTA)


if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    # if args.mode != "train":
    #     cfg.merge_from_list(['INFER.TTA', args.tta])
    # if args.debug:
    #     opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
    #     cfg.merge_from_list(opts)
    cfg.freeze()

    for _dir in ["WEIGHTS", "OUTPUTS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])

    setup_logging(args, cfg)
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)
