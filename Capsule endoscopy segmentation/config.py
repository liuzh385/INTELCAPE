from yacs.config import CfgNode as CN

_C = CN()

_C.EXP = "exp1"  # Experiment name
_C.COMMENT = "dice+no_aug+leaky_value"  # comment for tensorboard
_C.DEBUG = False

_C.INFER = CN()
_C.INFER.TTA = False

_C.MODEL = CN()
_C.MODEL.DICE_LOSS = False
_C.MODEL.BCE_LOSS = False
_C.MODEL.CE_LOSS = False
_C.MODEL.L1_LOSS = False
_C.MODEL.L1_DICE = False
_C.MODEL.WEIGHT = ""
_C.MODEL.WEIGHT_PRETRAIN = ""
_C.MODEL.WEIGTH = 1.0
_C.MODEL.MAIN_CHANNEL = 4
_C.MODEL.DROPOUT = 0.2
_C.MODEL.INIT_CHANNEL = 4
_C.MODEL.ADAPTIVE = True
_C.MODEL.FIXED = True

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 42
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O0"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 2

# RJ
# _C.DIRS = CN()
# _C.DIRS.DATA = "/mnt/minio/node77/liuzheng/RJ/Data/RJ_mp4"
# _C.DIRS.DATA_CSV = "/mnt/minio/node77/liuzheng/RJ/Data/csv/RG_Frame.csv"
# _C.DIRS.WEIGHTS = "/mnt/minio/node77/liuzheng/RJ/weights_pre"
# _C.DIRS.OUTPUTS = "/mnt/minio/node77/liuzheng/RJ/output"
# _C.DIRS.LOGS = "/mnt/minio/node77/liuzheng/RJ/log"
# _C.DIRS.TEST = "./test/"

# SN
_C.DIRS = CN()
_C.DIRS.DATA = "/mnt/minio/node77/liuzheng/SN/Video/SN_avi"
_C.DIRS.DATA_CSV = "/mnt/minio/node77/liuzheng/SN/Data/SN_Frame.csv"
_C.DIRS.WEIGHTS = "/mnt/minio/node77/liuzheng/RJ/weights_pre"
_C.DIRS.OUTPUTS = "/mnt/minio/node77/liuzheng/RJ/output"
_C.DIRS.LOGS = "/mnt/minio/node77/liuzheng/RJ/log"
_C.DIRS.TEST = "/mnt/minio/node77/liuzheng/RJ/code02_2/test"

_C.DATA = CN()
_C.DATA.INP_CHANNELS = 1
_C.DATA.SEG_CLASSES = 1
_C.DATA.WINDOW_CENTER = 700
_C.DATA.WINDOW_WIDTH = 2100
_C.DATA.MG = False
_C.DATA.NAME = "LNDB"
_C.DATA.NUM = 1
_C.DATA.STEP = 1
_C.DATA.RESIZE = False
_C.DATA.SEG = "None"
_C.DATA.CROP = 128
_C.DATA.NPY = False
_C.DATA.JPG = False

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 2
_C.OPT.BASE_LR = 1e-4
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0

_C.METRIC = CN()
# if true, the output will be computed by sigmoid, if false -> softmax
_C.METRIC.SIGMOID = True
_C.METRIC.THRESHOLD = 0.5

_C.TRAIN = CN()
_C.TRAIN.CSV = ""
_C.TRAIN.FOLD = 0
_C.TRAIN.MODEL = "unet_CT_dsv_3D"  # Model name
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 1  # switch to 32 if train on server
_C.TRAIN.DROPOUT = 0.0
_C.TRAIN.AUGMENTATION = False
_C.TRAIN.EXCHANGE = 1
_C.TRAIN.CROP = (160, 192, 128)
_C.TRAIN.NUM_STEPS = 20

_C.VAL = CN()
_C.VAL.CSV = ""
_C.VAL.FOLD = 0
_C.VAL.BATCH_SIZE = 1  # switch to 32 if train on server
_C.VAL.EXCHANGE = 1
_C.VAL.CROP = (160, 192, 128)
_C.VAL.ITEM = 10000

_C.CONST = CN()

_C.TEST = CN()
_C.TEST.CSV = ""
_C.TEST.FOLD = 0
_C.TEST.BATCH_SIZE = 1
_C.TEST.THRESHOLD = 0.5
_C.TEST.PRED = 0
_C.TEST.TIMES = 0
_C.TEST.CAM = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
# CUDA_VISIBLE_DEVICES=2 python main.py --config '/home/zhaoxinkai/code02/expconfigs/exp_sample_1080.yaml'
