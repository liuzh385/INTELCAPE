import sys
import argparse
import logging
import torch.optim as optim
import torch.nn as nn

from config import get_cfg_defaults
from models import *
from datasets import get_dataset, get_debug_dataset, get_test
from lr_scheduler import LR_Scheduler
from helpers import setup_determinism
from losses import BinaryDiceLoss, SoftDiceLoss, get_loss
import warnings
warnings.filterwarnings("ignore")
# torch.cuda.set_device(7)
# CUDA_VISIBLE_DEVICES=1
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
    parser.add_argument("--tta", action="store_true",
                        help="enable tta infer")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="enable debug mode for test")

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
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
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

    # Define Loss and Optimizer
    # train
    train_criterion = get_loss(cfg)
    valid_criterion = get_loss(cfg)

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

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        if cfg.MODEL.L1_DICE:
            train_criterion = [train_criterion[0].cuda(), train_criterion[1].cuda()]
            valid_criterion = [valid_criterion[0].cuda(), valid_criterion[1].cuda()]
        else:
            train_criterion = train_criterion.cuda()
            valid_criterion = valid_criterion.cuda()
    print(train_criterion)
    print(valid_criterion)

    print(f"=> loading checkpoint")
    weight = "/mnt/minio/node77/liuzheng/RJ/code02_2/weights/exp03_f0_resTFE_gau_ResNet_TFE_gau_fold0.pth"
    model.load_state_dict(torch.load(weight)["state_dict"])
    # ckpt = torch.load(weight, "cpu")
    # if not args.finetune:
    #     print("resuming optimizer ...")
    #     optimizer.load_state_dict(ckpt.pop('optimizer'))
    #     start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
    #     logging.info(
    #         f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
    
    
    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            print(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            model.load_state_dict({"resnet." + name: value for name, value in ckpt.pop('state_dict').items()},
                                  strict=False)
            # for p in model.resnet.parameters():
            #     p.requires_grad = False

            # print(model.densenet)
            # model.load_state_dict({"densenet." + name: value for name, value in ckpt.pop('state_dict').items()},
            #                       strict=False)
            # for p in model.densenet.features.parameters():
            #     p.requires_grad = False

            # compare
            # model.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
            #                       strict=True)
            # for p in model.features.parameters():
            #     p.requires_grad = False

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
    train_loaders = get_dataset('train', cfg)
    valid_loaders = get_dataset('valid', cfg)
    test_loaders = get_dataset('test', cfg)
    # test_loaders = None

    if cfg.DEBUG:
        train_loader = get_debug_dataset('train', cfg)
        valid_loader = get_debug_dataset('valid', cfg)

    scheduler = LR_Scheduler("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS, iters_per_epoch=len(train_loaders[0]),
                             warmup_epochs=cfg.OPT.WARMUP_EPOCHS)

    if args.mode == "train":
        if 'RL' in cfg.TRAIN.MODEL:
            train_RL_loop(logging.info, cfg, model, train_loaders, valid_loaders, train_criterion, valid_criterion,
                       optimizer, scheduler, start_epoch, best_metric, test_loaders)
        else:
            train_loop(logging.info, cfg, model, train_loaders, valid_loaders, train_criterion, valid_criterion,
                       optimizer, scheduler, start_epoch, best_metric, test_loaders)
            test_model(logging.info, cfg, model, test_loaders, weight=None, tta=cfg.INFER.TTA)
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_criterion, valid_loaders, tta=cfg.INFER.TTA)
    else:
        test_model(logging.info, cfg, model, test_loaders, weight=cfg.MODEL.WEIGHT, tta=cfg.INFER.TTA)
        # cfg_epoch = cfg.clone()
        # cfg_epoch.defrost()
        # cfg_epoch.DIRS.TEST = os.path.join(cfg_epoch.DIRS.TEST, "train_epoch")
        # test_model(logging.info, cfg_epoch, model, train_loader, weight=None, tta=cfg.INFER.TTA)


if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.mode != "train":
        cfg.merge_from_list(['INFER.TTA', args.tta])
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
    cfg.freeze()

    for _dir in ["WEIGHTS", "OUTPUTS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])

    setup_logging(args, cfg)
    setup_determinism(cfg.SYSTEM.SEED)
    print(cfg.DATA.CROP)
    main(args, cfg)
