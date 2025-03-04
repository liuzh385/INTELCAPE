import torch
import torch.nn as nn
import torchvision


class GaussianFilter(nn.Module):

    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)
        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):

    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):

    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad, gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img_lf = self.filter_low(img)
        img_hf = img - img_lf
        if self.normalize:
            return 0.5 + img_hf * 0.5, img_lf
        else:
            return img_hf, img_lf

class Resnet_gau(nn.Module):
    
    def __init__(self, resnet='resnet34'):
        super(Resnet_gau, self).__init__()
        
        self.gau_split = FilterHigh()
        if resnet == 'resnet18':
            backbone = torchvision.models.resnet18(pretrained=True)
            self.rgb_net = nn.Sequential(*(list(backbone.children())[:-1]))
            backbone = torchvision.models.resnet18(pretrained=True)
            self.gau_net = nn.Sequential(*(list(backbone.children())[:-1]))
        elif resnet == 'resnet34':
            backbone = torchvision.models.resnet34(pretrained=True)
            self.rgb_net = nn.Sequential(*(list(backbone.children())[:-1]))
            backbone = torchvision.models.resnet34(pretrained=True)
            self.gau_net = nn.Sequential(*(list(backbone.children())[:-1]))
        weights = "/mnt/minio/node77/liuzheng/RJ/code02_2/weights/best_exp02_100_f0_res_gau_ResNet_gau_fold0.pth"
        ckpt = torch.load(weights, "cpu")
        resnet.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
            strict=True)
        fc_features = backbone.fc.in_features
        self.fc1 = torch.nn.Linear(fc_features * 2, fc_features)
        self.fc2 = torch.nn.Linear(fc_features, 3)
        
    def forward(self, img):
        # print(img.shape)
        gau_img, _ = self.gau_split(img)
        rgb_feat, gau_feat = self.rgb_net(img).squeeze(-1).squeeze(-1), self.gau_net(gau_img).squeeze(-1).squeeze(-1)
        # print(rgb_feat.shape, gau_feat.shape)
        # input()
        # ret_feat = self.fc1(torch.cat((rgb_feat, gau_feat), -1))
        # ret_feat = self.fc2(ret_feat)
        
        return self.fc2(self.fc1(torch.cat((rgb_feat, gau_feat), -1)))