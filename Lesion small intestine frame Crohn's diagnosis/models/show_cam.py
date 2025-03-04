import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import os
import torch.nn.functional as F
from pylab import *


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # print(name, module, x.size())
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            # print(x.size())
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            elif "classifier" in name.lower():
                x = F.relu(x, inplace=True)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                x = module(x)
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, output


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam(cfg, model, img, save_name, item_name, target=None):
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    save_root = cfg.DIRS.TEST

    # args = get_args()
    use_cuda = True

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)

    if cfg.TRAIN.MODEL == "ResNet":
        grad_cam = GradCam(model=model, feature_module=model.layer4,
                           target_layer_names=["2"], use_cuda=use_cuda)
    elif cfg.TRAIN.MODEL == "DenseNet":
        grad_cam = GradCam(model=model, feature_module=model.features, target_layer_names=["norm5"],
                           use_cuda=use_cuda)

    # img = cv2.imread(args.image_path, 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)
    input = img
    img = img.squeeze().permute(1, 2, 0).cpu()

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = 0
    mask, output = grad_cam(input, target_index)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cam = heatmap + np.float32(img)
    cam_0 = cam / np.max(cam)

    target_index = 1
    mask, output = grad_cam(input, target_index)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cam = heatmap + np.float32(img)
    cam_1 = cam / np.max(cam)

    target_index = 2
    mask, output = grad_cam(input, target_index)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cam = heatmap + np.float32(img)
    cam_2 = cam / np.max(cam)

    os.makedirs(os.path.join(save_root, save_name), exist_ok=True)
    # cv2.imwrite(os.path.join(save_root, save_name, item_name + "_cam.jpg"), np.uint8(255 * cam))
    cam = np.zeros((640, 640, 3))
    cam[:320, :320, :] = np.float32(img) / 255
    cam[:320, 320:, :] = cam_0
    cam[320:, :320, :] = cam_1
    cam[320:, 320:, :] = cam_2
    cv2.imwrite(os.path.join(save_root, save_name, item_name + "_cam_cv2.jpg"), np.uint8(255 * cam))
    # plt
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    cam_0 = cv2.cvtColor(cam_0, cv2.COLOR_BGR2RGB)
    cam_1 = cv2.cvtColor(cam_1, cv2.COLOR_BGR2RGB)
    cam_2 = cv2.cvtColor(cam_2, cv2.COLOR_BGR2RGB)
    # print(output)

    fig = plt.figure(figsize=(4, 4), dpi=80)
    plt.subplot(221)
    plt.imshow(img)
    plt.title(f'img_{target[0].item()}')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(np.uint8(255 * cam_0))
    plt.title(f'HM_Before_{output[0][0].item():.2f}')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(np.uint8(255 * cam_1))
    plt.title(f'HM_SI_{output[0][1].item():.2f}')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(np.uint8(255 * cam_2))
    plt.title(f'HM_After_{output[0][2].item():.2f}')
    plt.axis('off')
    # plt.show()
    pred = output[0].clone().cpu().detach().numpy()
    pred_top = np.argmax(pred)
    flag = (int(pred_top) == int(target[0].item()))
    plt.savefig(os.path.join(save_root, save_name, item_name + f"_cam_pred{pred_top}_gt{target[0].item()}_{flag}.jpg"))

    print(os.path.join(save_root, save_name, item_name + "_cam.jpg"))

    return output.detach()
