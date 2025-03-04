from __future__ import print_function
import os
import shutil
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

import random
import copy
import imageio
import string

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def _initialize_weights(module):

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def save_checkpoint(state, is_best, root, filename):

    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'best_' + filename))


def cutmix_data(inputs, targets, alpha=1.):

    bsize, _, h, w = inputs.shape
    shuffled_idxs = torch.randperm(bsize).cuda()
    
    inputs_s = inputs[shuffled_idxs]
    lamb = np.random.beta(alpha, alpha)
    
    rx = np.random.randint(w)
    ry = np.random.randint(h)
    cut_ratio = np.sqrt(1. - lamb)
    rw = np.int(cut_ratio * w)
    rh = np.int(cut_ratio * h)

    x1 = np.clip(rx - rw // 2, 0, w)
    x2 = np.clip(rx + rw // 2, 0, w)
    y1 = np.clip(ry - rh // 2, 0, h)
    y2 = np.clip(ry + rh // 2, 0, h)
    
    inputs[:, :, x1:x2, y1:y2] = inputs_s[:, :, x1:x2, y1:y2]
    # adjust lambda to exactly match pixel ratio
    lamb = 1 - ((x2 - x1) * (y2 - y1) / (inputs.size()[-1] * inputs.size()[-2]))
    return inputs, targets, targets[shuffled_idxs], lamb

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
    
def mixup_criterion(criterion, pred, y_a, y_b, lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def DICE(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.sum(result* target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU

def IOU(output, target, SMOOTH=1e-6):
    # output = output.squeeze(1)
    output = output.max(axis=1)[1]
    output = output.view(output.size(0), -1)
    target = target.view(target.size(0), -1)
    
    intersect = output*target
    union = ((output + target) > 0).type(intersect.type())
    iou = (intersect.sum(1) + SMOOTH)/(union.sum(1) + SMOOTH)

    return iou.mean()

def apply_softmax(pred):
    return F.softmax(pred, dim=1)
    
def apply_sigmoid(pred):
    return F.sigmoid(pred)

# ==========================================================


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, prob=0.5):
    # augmentation by flipping
    cnt = 2
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        cnt = cnt - 1

    return x


def data_augmentation2(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 32
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        block_noise_size_z = random.randint(1, img_deps // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        noise_z = random.randint(0, img_deps - block_noise_size_z)
        window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y,
                 noise_z:noise_z + block_noise_size_z,
                 ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 2
    while cnt > 0 and random.random() < 0.65:
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
        block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                               block_noise_size_y,
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x


def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
    x[:,
    noise_x:noise_x + block_noise_size_x,
    noise_y:noise_y + block_noise_size_y,
    noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                            noise_y:noise_y + block_noise_size_y,
                                            noise_z:noise_z + block_noise_size_z]
    cnt = 2
    while cnt > 0 and random.random() < 0.65:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt -= 1
    return x


def generate_pair(img, batch_size, config, status="test"):
    img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):

            # Autoencoder
            x[n] = copy.deepcopy(y[n])

            # Flip
            x[n] = data_augmentation(x[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)

            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)

            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        # Save sample images module
        if config.save_samples is not None and status == "train" and random.random() < 0.01:
            n_sample = random.choice([i for i in range(config.batch_size)])
            sample_1 = np.concatenate(
                (x[n_sample, 0, :, :, 2 * img_deps // 6], y[n_sample, 0, :, :, 2 * img_deps // 6]), axis=1)
            sample_2 = np.concatenate(
                (x[n_sample, 0, :, :, 3 * img_deps // 6], y[n_sample, 0, :, :, 3 * img_deps // 6]), axis=1)
            sample_3 = np.concatenate(
                (x[n_sample, 0, :, :, 4 * img_deps // 6], y[n_sample, 0, :, :, 4 * img_deps // 6]), axis=1)
            sample_4 = np.concatenate(
                (x[n_sample, 0, :, :, 5 * img_deps // 6], y[n_sample, 0, :, :, 5 * img_deps // 6]), axis=1)
            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
            final_sample = final_sample * 255.0
            final_sample = final_sample.astype(np.uint8)
            file_name = ''.join(
                [random.choice(string.ascii_letters + string.digits) for n in range(10)]) + '.' + config.save_samples
            imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield x

if __name__ == "__main__":
    out = torch.rand((2,2,80,80,80), dtype=torch.float)
    target = torch.zeros((2,80,80,80), dtype=torch.bool)
    iou = dice_score(out, target)
    print(iou)