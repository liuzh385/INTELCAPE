
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim
from lion_pytorch import Lion

from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer
from util import string_contains_any
import wsol
import wsol.method


def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
        "Crohn15to23": 2
    }

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.','aggregator', 'classifier'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.cur_epoch = 0
        self.args = get_configs()
        # seed = self.args.seed
        # torch.manual_seed(seed)            # 为CPU设置随机种子
        # torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
        # torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
        # random.seed(seed)
        # np.random.seed(seed)
        # print(self.args.seed)
        #set_random_seed(self.args.seed)
        
        print(self.args)
        self.reporter = self.args.reporter
        self.model = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            splits=('test',))

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold,
            num_head = self.args.num_head)
        model = model.cuda()
        print(model)
        return model

    def evaluate(self, epoch, split):
        print("Evaluate split {}".format(split))
        self.model.eval()

        cam_computer = CAMComputer(
            model=self.model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
            wsol_method = self.args.wsol_method,
            target_layer = self.args.target_layer,
            is_vis = self.args.is_vis,
            eval_type = self.args.eval_type,
            just_gen_cams=True
        )
        cam_computer.gen_cams_and_boxes()


def main():
    trainer = Trainer()

    
    print("===========================================================")
    
    if "test" in trainer.args.mode:
        checkpoint = torch.load(trainer.args.check_path)
        if trainer.args.dataset_name == "ILSVRC" or trainer.args.dataset_name == "OpenImages":
            from wsol.util import replace_layer
            checkpoint['state_dict'] = replace_layer(checkpoint['state_dict'], 'extractor_A', 'aggregator_A')
            checkpoint['state_dict'] = replace_layer(checkpoint['state_dict'], 'extractor_B', 'aggregator_B')
        trainer.model.load_state_dict(checkpoint['state_dict'], strict=True)

        trainer.evaluate(trainer.args.epochs, split='test')

        return


if __name__ == '__main__':
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    main()
