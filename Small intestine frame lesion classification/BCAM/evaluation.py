"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRMaskANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
from skimage.segmentation import mark_boundaries
import cv2
import numpy as np
import os
from os.path import join as ospj
import torch.utils.data as torchdata

from config import str2bool
from data_loaders import configure_metadata
from data_loaders import get_image_ids
from data_loaders import get_bounding_boxes
from data_loaders import get_image_sizes
from data_loaders import get_mask_paths
from util import check_scoremap_validity
from util import check_box_convention
from util import t2n

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0



def generate_vis(p, img):
    # All the input should be numpy.array 
    # img should be 0-255 uint8

    C = 1
    H, W = p.shape

    prob = p

    prob[prob<=0] = 1e-7

    def ColorCAM(prob, img):
        C = 1
        H, W = prob.shape
        colorlist = []
        colorlist.append(color_pro(prob,img=img,mode='chw'))
        CAM = np.array(colorlist)/255.0
        return CAM

    #print(prob.shape, img.shape)
    CAM = ColorCAM(prob, img)
    #print(CAM.shape)
    return CAM[0, :, :, :]

def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))	
	return color

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float64)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float64) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    # newbox_x0 = max(box_x0-20, 0) * new_image_w / (image_w-40)
    # newbox_y0 = max(box_y0-20, 0) * new_image_h / (image_h-40)
    # newbox_x1 = min(box_x1-20, 320) * new_image_w / (image_w-40)
    # newbox_y1 = min(box_y1-20, 320) * new_image_h / (image_h-40)
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, image_ids):
        self.scoremap_path = scoremap_path
        self.predict_path = os.path.join('/'.join(scoremap_path.split('/')[:-1]), "classes")
        self.target_path = os.path.join('/'.join(scoremap_path.split('/')[:-1]), "targets")
        self.image_ids = image_ids

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, image_id + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        cam[cam > 1] = 1
        predict = np.load(os.path.join(self.predict_path, image_id + '.npy'))
        target = np.load(os.path.join(self.target_path, image_id + '.npy'))
        return cam, image_id, predict, target

    def __len__(self):
        return len(self.image_ids)


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, dataset_name, split, cam_threshold_list,
                 iou_threshold_list, mask_root, multi_contour_eval):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.multi_contour_eval = multi_contour_eval

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

        ###
        self.num_correct_top1 = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.best_pred_boxes = {}
        self.num_correct_ois =  {iou_threshold: np.zeros(1)
             for iou_threshold in self.iou_threshold_list}
        self.num_correct_top1_ois =  {iou_threshold: np.zeros(1)
             for iou_threshold in self.iou_threshold_list}
        ###

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremap, image_id, label, pred_label, image):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        ###
        best_iou_box_ind = np.argmax(multiple_iou, axis=0)
        self.best_pred_boxes[image_id] = boxes_at_thresholds[best_iou_box_ind]
        ###

        idx = 0
        sliced_multiple_iou = []
        max_iou_row_indexs = []
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            # ------add-------
            # max_iou_coordinates = np.unravel_index(multiple_iou[idx:idx + nr_box].argmax(), multiple_iou[idx:idx + nr_box].shape)
            # max_iou_row_indexs.append(idx + max_iou_coordinates[0])
            #-----------------
            idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]
            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1
            if label == pred_label:
                self.num_correct_top1[_THRESHOLD][correct_threshold_indices] += 1
            # -----add-----
            # selected_row_indices = np.array(max_iou_row_indexs)[correct_threshold_indices]
            # boxes = boxes_at_thresholds[selected_row_indices]
            # txt_file = f'/mnt/minio/node77/caiyinqi/BCAM-main/SingleObjectLocalization/test_log_Crohn15to23_prob/resnet34/threshold{_THRESHOLD}.txt'
            # with open(txt_file, 'a', encoding='gbk') as f:
            #     f.write(f'{image_id}\n')
            #     for box in boxes:
            #         array_str = np.array2string(box, separator=',')[1:-1]  # 去掉开头和结尾的方括号         
            #         f.write(f'{array_str}\n')
            # ------------
            
            ##OIS results###
            #print(best_iou_box_ind)
            #if multiple_iou[best_iou_box_ind[0]] > (_THRESHOLD/100):
            #    self.num_correct_ois[_THRESHOLD] += 1
            #    if label == pred_label:
            #        self.num_correct_top1_ois[_THRESHOLD] += 1

        self.cnt += 1

        #-----------------add--------------------
        # draw bounding box
        # img_np = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
        # img_np = np.int64(img_np * 255)
        # img_np[img_np > 255] = 255
        # img_np[img_np < 0] = 0
        # img_np = np.uint8(img_np)
        # img_np = img_np.transpose((1, 2, 0)).copy()
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
        # _, thr_gray_heatmap = cv2.threshold(
        #     src=scoremap_image,
        #     thresh=int(0.1 * np.max(scoremap_image)),
        #     maxval=255,
        #     type=cv2.THRESH_BINARY)
        # contours = cv2.findContours(
        #     image=thr_gray_heatmap,
        #     mode=cv2.RETR_TREE,
        #     method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
        # print(len(contours))
        # max_contour = max(contours, key=cv2.contourArea)
        # max_contour_area = cv2.contourArea(max_contour)
        # for contour in contours:
        #     if cv2.contourArea(contour) > 0.15 * max_contour_area:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         multiple_iou = calculate_multiple_iou(
        #             np.array([[x, y, x + w, y + h]]),
        #             np.array(self.gt_bboxes[image_id]))
        #         for i in range(multiple_iou.shape[1]):
        #             x0, y0, x1, y1 = self.gt_bboxes[image_id][i][:]
        #             img_np = cv2.rectangle(img_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
        #             # if multiple_iou[0][i] >= 0.1:
        #             img_np = cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # imgs_dir = '/mnt/minio/node77/caiyinqi/BCAM-main/SingleObjectLocalization/test_log_Crohn15to23_prob/resnet34_bas_acol_epo16/rect_jpg/'
        # if not os.path.exists(imgs_dir):
        #     os.makedirs(imgs_dir)
        # uni_cn_name = image_id.encode("unicode_escape")
        # uni_cn_name = uni_cn_name.decode().replace('test/', '').replace('\\', '')
        # img_file = f'/mnt/minio/node77/caiyinqi/BCAM-main/SingleObjectLocalization/test_log_Crohn15to23_prob/resnet34_bas_acol_epo16/rect_jpg/{uni_cn_name}'
        # cv2.imencode('.jpg', img_np)[1].tofile(img_file)
        #----------------------------------------

    
    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []
        #lists_acc = {}

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            #lists_acc[_THRESHOLD] = localization_accuracies
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc#, lists_acc

    def compute_top1(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct_top1[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc

    def compute_ois(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct_ois[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc

    def compute_top1_ois(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct_top1_ois[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc

    def compute_list(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []
        lists_acc = {}

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            lists_acc[_THRESHOLD] = localization_accuracies
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc, lists_acc

    def compute_top_list(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []
        lists_acc = {}

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct_top1[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            lists_acc[_THRESHOLD] = localization_accuracies
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc, lists_acc

def load_mask_image(file_path, resize_size):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    #print(file_path)
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask


def get_mask(mask_root, mask_paths, ignore_path):
    """
    Ignore mask is set as the ignore box region \setminus the ground truth
    foreground region.

    Args:
        mask_root: string.
        mask_paths: iterable of strings.
        ignore_path: string.

    Returns:
        mask: numpy.ndarray(size=(224, 224), dtype=np.uint8)
    """
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (_RESIZE_LENGTH, _RESIZE_LENGTH))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

    ignore_file = os.path.join(mask_root, ignore_path)
    ignore_box_mask = load_mask_image(ignore_file,
                                      (_RESIZE_LENGTH, _RESIZE_LENGTH))
    ignore_box_mask = ignore_box_mask > 0.5

    ignore_mask = np.logical_and(ignore_box_mask,
                                 np.logical_not(mask_all_instances))

    if np.logical_and(ignore_mask, mask_all_instances).any():
        raise RuntimeError("Ignore and foreground masks intersect.")

    return (mask_all_instances.astype(np.uint8) +
            255 * ignore_mask.astype(np.uint8))


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        if self.dataset_name != "OpenImages":
            raise ValueError("Mask evaluation must be performed on OpenImages.")

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])
        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float64)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float64)

    def accumulate(self, scoremap, image_id):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float64)
            image_id: string.
        """
        check_scoremap_validity(scoremap)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(np.float64)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(np.float64)

    def accumulate_vis(self, scoremap, image_id):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float64)
            image_id: string.
        """
        check_scoremap_validity(scoremap)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, bins_true = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(np.float64)

        gt_false_hist, bins_false = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(np.float64)


        #####
        num_gt_true = gt_true_hist.astype(np.float64).sum()
        tp = gt_true_hist.astype(np.float64)[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = gt_false_hist.astype(np.float64).sum()
        fp = gt_false_hist.astype(np.float64)[::-1].cumsum()
        tn = num_gt_false - fp

        mIoU = tp / (fp + fn + tp)

        index = np.argmax(mIoU)

        mIoU = np.max(mIoU)
        predict = scoremap.copy()
        predict[predict > 1 - bins_true[index]] = 1
        predict[predict != 1] = 0
        predict = np.int64(predict)
        np_gt = np.array(gt_mask)
        np_gt[np_gt == 255] = 0
        img = cv2.imread(os.path.join('/home/zl/WeaklySeg/Active_OIS/wsolevaluation-master/dataset', 'OpenImages', image_id))

        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_NEAREST)

        name = image_id.split('/')[-1][:-4]

        #print(np.unique(img))
        save_gt = mark_boundaries(img.copy(), np_gt, mode='thick', color=(255, 0, 0))
        save_gt[save_gt!=255] *= 255
        cv2.imwrite(os.path.join(self.vis_gt_dir, name) + ".jpg", save_gt)

        cam = generate_vis(scoremap.copy(), img.copy().transpose(2, 0, 1)).transpose(1, 2, 0)

        save_pred = mark_boundaries(cam.copy(), predict, mode='thick', color=(1, 0, 0))

        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(self.vis_pred_dir, name + "_%.4f.jpg"%(mIoU)), save_pred)
        #plt.imsave(os.path.join(self.save_dir, name + "_%.4f_cam.png"%(mIoU)), np.int64(255 * generate_vis(1 - scoremap, img.transpose(2, 0, 1)).transpose(1, 2, 0)))
        #####


    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        mIoU = tp / (fp + fn + tp)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        pIoU = np.max(mIoU) * 100

        self.BPs = precision * 100
        self.BRs = recall * 100
        self.mIoUs = mIoU * 100

        self.BPs = self.BPs[~np.isnan(self.BPs)]
        self.BRs = self.BRs[~np.isnan(self.BRs)]
        self.mIoUs = self.mIoUs[~np.isnan(self.mIoUs)]


        PxAP = auc

        #print("Mask AUC on split {}: {}".format(self.split, auc))
        return pIoU, PxAP

def _get_cam_loader(image_ids, scoremap_path):
    return torchdata.DataLoader(
        CamDataset(scoremap_path, image_ids),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True)


def evaluate_wsol(args, scoremap_root, metadata_root, mask_root, dataset_name, split,
                  multi_contour_eval, multi_iou_eval, iou_threshold_list,
                  cam_curve_interval=.001):
    """
    Compute WSOL performances of predicted heatmaps against ground truth
    boxes (CUB, ILSVRC) or masks (OpenImages). For boxes, we compute the
    gt-known box accuracy (IoU>=0.5) at the optimal heatmap threshold.
    For masks, we compute the area-under-curve of the pixel-wise precision-
    recall curve.

    Args:
        scoremap_root: string. Score maps for each eval image are saved under
            the output_path, with the name corresponding to their image_ids.
            For example, the heatmap for the image "123/456.JPEG" is expected
            to be located at "{output_path}/123/456.npy".
            The heatmaps must be numpy arrays of type np.float, with 2
            dimensions corresponding to height and width. The height and width
            must be identical to those of the original image. The heatmap values
            must be in the [0, 1] range. The map must attain values 0.0 and 1.0.
            See check_scoremap_validity() in util.py for the exact requirements.
        metadata_root: string.
        mask_root: string.
        dataset_name: string. Supports [CUB, ILSVRC, and OpenImages].
        split: string. Supports [train, val, test].
        multi_contour_eval:  considering the best match between the set of all
            estimated boxes and the set of all ground truth boxes.
        multi_iou_eval: averaging the performance across various level of iou
            thresholds.
        iou_threshold_list: list. default: [30, 50, 70]
        cam_curve_interval: float. Default 0.001. At which threshold intervals
            will the heatmaps be evaluated?
    Returns:
        performance: float. For CUB and ILSVRC, maxboxacc is returned.
            For OpenImages, area-under-curve of the precision-recall curve
            is returned.
    """
    print("Loading and evaluating cams.")
    meta_path = os.path.join(metadata_root, dataset_name, split)
    metadata = configure_metadata(meta_path)
    image_ids = get_image_ids(metadata)
    cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

    evaluator = {"OpenImages": MaskEvaluator,
                 "CUB": BoxEvaluator,
                 "Crohn15to23": BoxEvaluator,
                 "ILSVRC": BoxEvaluator
                 }[dataset_name](metadata=metadata,
                                 dataset_name=dataset_name,
                                 split=split,
                                 cam_threshold_list=cam_threshold_list,
                                 mask_root=ospj(mask_root, 'OpenImages'),
                                 multi_contour_eval=multi_contour_eval,
                                 iou_threshold_list=iou_threshold_list)

    vis_pred_dir = os.path.join(args.save_dir, args.check_name)
    if not os.path.exists(vis_pred_dir):
        os.mkdir(vis_pred_dir)
    vis_gt_dir = os.path.join(args.save_dir, "gt")
    if not os.path.exists(vis_gt_dir):
        os.mkdir(vis_gt_dir)

    evaluator.vis_pred_dir = vis_pred_dir
    evaluator.vis_gt_dir = vis_gt_dir

    cam_loader = _get_cam_loader(image_ids, scoremap_root)
    for cams, image_ids, predicts, targets in cam_loader:
        for cam, image_id, predict, target in zip(cams, image_ids, predicts, targets):
            if dataset_name == "OpenImages":
                evaluator.accumulate_vis(t2n(cam), image_id)
            else:
                evaluator.accumulate(t2n(cam), image_id, predict, target) 
    if dataset_name == "OpenImages":

        performance = evaluator.compute()
        print(performance)
        evaluator.v

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        write_row = []
        write_row.append(args.check_name)
        with open(os.path.join(args.save_dir, 'PC_lists.csv'), "a") as f:
            writer = csv.writer(f)
            write_row = write_row + evaluator.BPs.tolist()
            writer.writerow(write_row)

        write_row = []
        write_row.append(args.check_name)
        import csv
        with open(os.path.join(args.save_dir, 'BR_lists.csv'), "a") as f:
            writer = csv.writer(f)
            write_row = write_row + evaluator.BRs.tolist()
            writer.writerow(write_row)

        write_row = []
        write_row.append(args.check_name)
        with open(os.path.join(args.save_dir, 'IoU_lists.csv'), "a") as f:
            writer = csv.writer(f)
            write_row = write_row + evaluator.mIoUs.tolist()
            writer.writerow(write_row)
            
    else:
        performance, mba_list = evaluator.compute_list()
        performance_top, top_list = evaluator.compute_top_list()

        mba_ois = evaluator.compute_ois()
        top1_ois = evaluator.compute_top1_ois()

        print("MBA Performance: ", performance)
        print("Top Performance: ", performance_top)

        print("OIS MBA Performance: ", mba_ois)
        print("OIS Top Performance: ", top1_ois)

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        import csv
        for ind in iou_threshold_list:
            write_row = []
            write_row.append(args.check_name)
            with open(os.path.join(args.save_dir, 'mba_%s_lists.csv'%(str(ind))), "a") as f:
                writer = csv.writer(f)
                write_row = write_row + mba_list[ind].tolist()
                writer.writerow(write_row)

        for ind in iou_threshold_list:
            write_row = []
            write_row.append(args.check_name)
            with open(os.path.join(args.save_dir, 'top_%s_lists.csv'%(str(ind))), "a") as f:
                writer = csv.writer(f)
                write_row = write_row + top_list[ind].tolist()
                writer.writerow(write_row)

        if dataset_name != "ILSVRC" or (dataset_name == "ILSVRC" and split=="test"):
            vis_pred_dir = os.path.join(args.save_dir, args.check_name)
            if not os.path.exists(vis_pred_dir):
                os.mkdir(vis_pred_dir)
            vis_gt_dir = os.path.join(args.save_dir, "gt")
            if not os.path.exists(vis_gt_dir):
                os.mkdir(vis_gt_dir)
            for k, v in evaluator.original_bboxes.items():
                pred = evaluator.best_pred_boxes[k][0]
                gt = evaluator.gt_bboxes[k][0]
                img = cv2.imread(os.path.join(args.img_root, k))
                cam = np.load(os.path.join(args.scoremap_root, k +".npy"))
                #print(np.unique(cam))
                #print(cam.shape)
                v = v[0]
                img = cv2.resize(img, (_RESIZE_LENGTH, _RESIZE_LENGTH), interpolation=cv2.INTER_NEAREST)
                cam = generate_vis(cam.copy(), img.copy().transpose(2, 0, 1)).transpose(1, 2, 0)
                gt_bbox = cv2.rectangle(img.copy(), (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 4)
                cam_bbox = cv2.rectangle(cam.copy(), (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (1, 0, 0), 4)
                #bbox = cv2.rectangle(bbox, (pred[0], pred[1]), (pred[2], pred[3]), (0, 0, 255), 4)

                cur_score = calculate_multiple_iou(np.array(pred).reshape(1, 4), np.array(gt).reshape(1, 4))
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(vis_pred_dir, k.split('/')[-1][:-4] + "_%.2f.jpg"%(cur_score)), cam_bbox)
                cv2.imwrite(os.path.join(vis_gt_dir, k.split('/')[-1][:-4]) + ".jpg", gt_bbox)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str,
                        default='/mnt/nas/zhulei/datas/dataset/CUB',
                        help="The root folder for score maps to be evaluated.")
    parser.add_argument('--scoremap_root', type=str,
                        default='train_log/scoremaps/',
                        help="The root folder for score maps to be evaluated.")
    parser.add_argument('--metadata_root', type=str, default='metadata/',
                        help="Root folder of metadata.")
    parser.add_argument('--mask_root', type=str, default='/mnt/nas/zhulei/datas/dataset/',
                        help="Root folder of masks (OpenImages).")
    parser.add_argument('--dataset_name', type=str,
                        help="One of [CUB, ImageNet, OpenImages].")
    parser.add_argument('--split', type=str,
                        help="One of [val, test]. They correspond to "
                             "train-fullsup and test, respectively.")
    parser.add_argument('--cam_curve_interval', type=float, default=0.01,
                        help="At which threshold intervals will the score maps "
                             "be evaluated?.")
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])

    parser.add_argument('--save_dir', type=str, default="eval_log_CUB")
    parser.add_argument('--check_name', type=str, default="prob")

    args = parser.parse_args()
    evaluate_wsol(args = args, 
                  scoremap_root=args.scoremap_root,
                  metadata_root=args.metadata_root,
                  mask_root=args.mask_root,
                  dataset_name=args.dataset_name,
                  split=args.split,
                  cam_curve_interval=args.cam_curve_interval,
                  multi_contour_eval=args.multi_contour_eval,
                  multi_iou_eval=args.multi_iou_eval,
                  iou_threshold_list=args.iou_threshold_list,)


if __name__ == "__main__":
    main()
