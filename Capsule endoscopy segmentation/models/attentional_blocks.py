import torch
from torch import nn
from torch.nn import functional as F
from .utils_unet import init_weights


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels

        conv_nd = nn.Conv3d

        self.w_im = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0, bias=True)
        self.w_i = conv_nd(in_channels=self.in_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x, g, g2):
        m = F.sigmoid(self.w_im(g2))
        o = g + m * self.w_i(g2)

        return o


class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


if __name__ == '__main__':
    from torch.autograd import Variable

    mode_list = ['concatenation']

    for mode in mode_list:

        img = Variable(torch.rand(2, 16, 10, 10, 10))
        gat = Variable(torch.rand(2, 64, 4, 4, 4))
        net = GridAttentionBlock3D(in_channels=16, inter_channels=16, gating_channels=64, mode=mode, sub_sample_factor=(2,2,2))
        out, sigma = net(img, gat)
        print(out.size())
