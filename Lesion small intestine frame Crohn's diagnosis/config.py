from yacs.config import CfgNode as CN, CfgNode

_C: CfgNode = CN()

_C.EXP = "exp1"  # Experiment name
_C.COMMENT = "dice+no_aug+leaky_value"  # comment for tensorboard
_C.DEBUG = False

_C.INFER = CN()
_C.INFER.TTA = False

_C.MODEL = CN()
_C.MODEL.DICE_LOSS = False
_C.MODEL.BCE_LOSS = False
_C.MODEL.CE_LOSS = False
_C.MODEL.FOCAL_LOSS = False
_C.MODEL.L1_LOSS = False
_C.MODEL.L1_DICE = False
_C.MODEL.WEIGHT = ''
_C.MODEL.WEIGHT_PRETRAIN = ""
_C.MODEL.WEIGTH = 1.0
_C.MODEL.MAIN_CHANNEL = 4
_C.MODEL.DROPOUT = 0.2
_C.MODEL.INIT_CHANNEL = 4
_C.MODEL.ADAPTIVE = True
_C.MODEL.FIXED = True
_C.MODEL.IMAGENET_PRETRAIN = False
_C.MODEL.FEATURE_EXTRACTOR = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 42
_C.SYSTEM.FP16 = False
_C.SYSTEM.OPT_L = "O0"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 2

_C.DIRS = CN()
_C.DIRS.DATA = "/data2/zhaoxinkai/Crohn_avi"
_C.DIRS.DATA_CSV = "./data/crohn2016_time_flag.csv"
_C.DIRS.DATA_SEQ = "./seq/samples/"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"
_C.DIRS.TEST = "./test/"

_C.DATA = CN()
_C.DATA.INP_CHANNELS = 1
_C.DATA.SEG_CLASSES = 1
_C.DATA.WINDOW_CENTER = 700
_C.DATA.WINDOW_WIDTH = 2100
_C.DATA.NAME = "LNDB"
_C.DATA.NUM = 1
_C.DATA.STEP = 1
_C.DATA.RESIZE = False
_C.DATA.SEG = "None"
_C.DATA.SIZE = 320
_C.DATA.NPY = False
_C.DATA.JPG = False
_C.DATA.UNLABELED_WEIGHT = 1
_C.DATA.DROPOUT = 0.5

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 2
_C.OPT.BASE_LR = 1e-3
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
_C.TRAIN.VIT_NAME = ''
_C.TRAIN.NUM_CLASS = 3
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
_C.VAL.EPOCH = 1

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

