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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")

    args = parser.parse_args()
    args.mode = "test"

    return args


def setup_logging(cfg):
    if not os.path.isdir(cfg.DIRS.LOGS):
        os.mkdir(cfg.DIRS.LOGS)

    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(
        os.path.join(cfg.DIRS.LOGS, f'{cfg.EXP}_{cfg.TRAIN.MODEL}_test5fold_fold{cfg.TRAIN.FOLD}.log'),
        mode='a'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info(f'===============================')
    logging.info(f'\n\nStart with config {cfg}')


def main(cfg):
    logging.info(f"=========> {cfg.EXP} <=========")

    # Declare variables
    for i in range(5):
        cfg_fold = cfg.clone()
        cfg_fold.defrost()
        cfg_fold.MODEL.WEIGHT_PRETRAIN = cfg_fold.MODEL.WEIGHT_PRETRAIN.replace('X', f'{i}')
        cfg_fold.MODEL.WEIGHT = cfg_fold.MODEL.WEIGHT.replace('X', f'{i}')
        cfg_fold.DIRS.TEST = cfg_fold.DIRS.TEST.replace('X', f'{i}')
        cfg_fold.TEST.FOLD = i

        model = get_model(cfg_fold)
        model = model.cuda()
        if cfg_fold.SYSTEM.MULTI_GPU:
            model = nn.DataParallel(model)

        # Load data
        test_loaders = get_dataset('test', cfg_fold)

        cfg_epoch = cfg_fold.clone()
        cfg_epoch.defrost()
        cfg_epoch.TEST.PRED = 0
        test_model(logging.info, cfg_epoch, model, test_loaders, weight=cfg_epoch.MODEL.WEIGHT, tta=cfg_epoch.INFER.TTA)
        cfg_epoch = cfg_fold.clone()
        cfg_epoch.defrost()
        cfg_epoch.TEST.PRED = 2
        test_model(logging.info, cfg_epoch, model, test_loaders, weight=cfg_epoch.MODEL.WEIGHT, tta=cfg_epoch.INFER.TTA)


if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()

    cfg.merge_from_file(args.config)
    cfg.freeze()

    setup_logging(cfg)
    setup_determinism(cfg.SYSTEM.SEED)
    main(cfg)
