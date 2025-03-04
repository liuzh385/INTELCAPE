import math
import torch.nn as nn
from .utils_unet import UnetConv3, UnetUp3, init_weights, UnetUp3_CT, UnetDsv3
import torch.nn.functional as F
import torch
import warnings
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair, _triple

warnings.filterwarnings("ignore")


class unet_3D(nn.Module):

    def __init__(self, n_classes=1, feature_scale=4, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv, is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv, is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv, is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print('c1', conv1.size(), maxpool1.size())

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print('c2', conv2.size(), maxpool2.size())

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print('c3', conv3.size(), maxpool3.size())

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # print('c4', conv4.size(), maxpool4.size())

        center = self.center(maxpool4)
        # print('center', center.size())

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        # final = final.squeeze(1)
        # print('fs', final.size())
        return F.sigmoid(final)
        # return self.apply_argmax_softmax(final)

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)

        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)

        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        c4d = self.dropout(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)
        uout = F.sigmoid(uout)

        return uout


class GaborConv3d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(GaborConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _triple(0), groups, bias, padding_mode)
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        self.z0 = torch.ceil(torch.Tensor([self.kernel_size[2] / 2]))[0]

    def forward(self, input):
        z, y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
                                  torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1]),
                                  torch.linspace(-self.z0 + 1, self.z0, self.kernel_size[2])])
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        weight = torch.empty(self.weight.shape, requires_grad=False).cuda()
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j]
                psi = self.psi[i, j].expand_as(y)

                R = self.rotation(theta)
                rotx = z * R[2, 0] + y * R[2, 1] + x * R[2, 2]
                roty = z * R[1, 0] + y * R[1, 1] + x * R[1, 2]
                rotz = z * R[0, 0] + y * R[0, 1] + x * R[0, 2]

                g = torch.zeros(y.shape)

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2 + rotz ** 2) / (sigma + 1e-3) ** 2))  # error
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g
        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def rotation(self, theta):
        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(theta), -torch.sin(theta)],
                            [0, torch.sin(theta), torch.cos(theta)]
                            ])

        R_y = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                            [0, 1, 0],
                            [-torch.sin(theta), 0, torch.cos(theta)]
                            ])

        R_z = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0],
                            [0, 0, 1]
                            ])

        R = torch.mm(R_z, torch.mm(R_y, R_x))  # G02
        # R = torch.mm(R_x, R_z)  # G01

        return R


class unet_3D_Gabor(nn.Module):

    def __init__(self, n_classes=1, feature_scale=4, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(unet_3D_Gabor, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.gaborConv = GaborConv3d(self.in_channels, 32, kernel_size=11, stride=1, padding=5)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(32, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv, is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv, is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv, is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        inputs = self.gaborConv(inputs)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print('c1', conv1.size(), maxpool1.size())

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print('c2', conv2.size(), maxpool2.size())

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print('c3', conv3.size(), maxpool3.size())

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # print('c4', conv4.size(), maxpool4.size())

        center = self.center(maxpool4)
        # print('center', center.size())

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        # final = final.squeeze(1)
        # print('fs', final.size())
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class unet_CT_dsv_3D(nn.Module):

    def __init__(self, n_classes, feature_scale=4, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(unet_CT_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3, 3, 3),
                               padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3, 3, 3),
                                padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes * 4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))

        # squeeze for bce loss
        final = final.squeeze(1)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=4):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


if __name__ == "__main__":
    model = unet_3D()
    print(model)
    Input = torch.rand((1, 1, 80, 80, 80))
    out = model(Input)
    # model.apply_argmax_softmax(out)
    print(out.shape)
