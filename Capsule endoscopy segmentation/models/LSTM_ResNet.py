import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import torch
from .resnet import *
from .densenet import *
from .res_gau import *
import logging

class DoNothing(nn.Module):

    def forward(self, x):
        return x
    
    
class Resnet_gau_TFE(nn.Module):
    
    def __init__(self, resnet='resnet34'):
        super(Resnet_gau_TFE, self).__init__()
        
        self.gau_split = FilterHigh()
        if resnet == 'resnet18':
            backbone = torchvision.models.resnet18(pretrained=False)
            self.rgb_net = nn.Sequential(*(list(backbone.children())[:-1]))
            backbone = torchvision.models.resnet18(pretrained=False)
            self.gau_net = nn.Sequential(*(list(backbone.children())[:-1]))
        elif resnet == 'resnet34':
            backbone = torchvision.models.resnet34(pretrained=False)
            self.rgb_net = nn.Sequential(*(list(backbone.children())[:-1]))
            backbone = torchvision.models.resnet34(pretrained=False)
            self.gau_net = nn.Sequential(*(list(backbone.children())[:-1]))
        fc_features = backbone.fc.in_features
        self.fc1 = torch.nn.Linear(fc_features * 2, fc_features)
        self.fc2 = torch.nn.Linear(fc_features, 3)
        
    def forward(self, img):
        # logging.info(img.shape)
        gau_img, _ = self.gau_split(img)
        # rgb_feat, gau_feat = self.rgb_net(img).squeeze(-1).squeeze(-1), self.gau_net(gau_img).squeeze(-1).squeeze(-1)
        rgb_feat = self.rgb_net(img).squeeze(-1).squeeze(-1)
        gau_feat = self.gau_net(gau_img).squeeze(-1).squeeze(-1)
        # logging.info(rgb_feat.shape, gau_feat.shape)
        # input()
        ret_feat = self.fc1(torch.cat((rgb_feat, gau_feat), -1))
        # ret_feat = self.fc2(ret_feat)
        
        # return self.fc2(self.fc1(torch.cat((rgb_feat, gau_feat), -1)))
        return ret_feat


class LSTMResNet(nn.Module):

    def __init__(self, output_size, pretrained=True):
        super(LSTMResNet, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        # fc_nums = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(fc_nums, 3)

        # self.fc_features = self.resnet.fc.in_features
        # self.resnet.fc = DoNothing()
        # self.lstm1 = nn.LSTM(self.fc_features, self.fc_features, 2, bidirectional=True)
        # self.lstm2 = nn.LSTM(self.fc_features * 2, self.fc_features, 2, bidirectional=True)

        self.num = 5
        self.fc_features = 3
        self.lstm = nn.LSTM(self.fc_features, self.fc_features, 2, bidirectional=True)

        self.fc = nn.Linear(self.fc_features * self.num * 2, 3)

    def forward(self, frames):
        # print('frames', frames.size())
        batch_size = frames.size()[0]
        size_a = frames.size()[-2]
        size_b = frames.size()[-1]
        frames = frames.view(-1, 3, size_a, size_b)
        output = self.resnet(frames)
        # print('output', output.size())
        output = output.view(batch_size, -1, self.fc_features)
        # print('output', output.size())

        # ans, _ = self.lstm1(output)
        # ans, _ = self.lstm2(ans)

        ans, _ = self.lstm(output)

        ans = ans.view(batch_size, self.fc_features * 2 * self.num)
        ans = self.fc(ans)
        # print('ans', ans.size())

        return ans


class TransformerResNet(nn.Module):

    def __init__(self, output_size, pretrained):
        super(TransformerResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.fc_features = self.resnet.fc.in_features
        self.resnet.fc = DoNothing()
        self.tf = TransformerEncoderLayer(d_model=512, nhead=8)
        # self.tf2 = TransformerEncoderLayer(d_model=512, nhead=8)
        self.fc = nn.Linear(self.fc_features * 5, output_size)

    def forward(self, frames):
        # print('frames', frames.size())
        batch_size = frames.size()[0]
        size_a = frames.size()[-2]
        size_b = frames.size()[-1]
        frames = frames.view(-1, 3, size_a, size_b)
        output = self.resnet(frames)
        # print('output', output.size())
        output = output.view(batch_size, -1, self.fc_features)
        # print('output', output.size())
        ans = self.tf(output)
        # ans = self.tf2(ans)
        ans = ans.view(batch_size, self.fc_features * 5)
        ans = self.fc(ans)
        # print('ans', ans.size())

        return ans


# 2D CNN encoder using ResNet-18 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, cfg, fc_hidden1=512, fc_hidden2=512, drop_p=0.1, CNN_embed_dim=300):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.fixed = cfg.MODEL.FIXED

        resnet = resnet34(pretrained=False)
        # resnet = resnet18(pretrained=False)
        fc_features = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(fc_features, 3)
        if cfg.MODEL.WEIGHT_PRETRAIN != "":
            ckpt = torch.load(cfg.MODEL.WEIGHT_PRETRAIN, "cpu")
            resnet.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
                                   strict=True)
            if self.fixed is True:
                for p in resnet.parameters():
                    p.requires_grad = False
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01, )
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01, )
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        # print(x_3d.size())
        for t in range(x_3d.size(1)):
            # ResNet CNN
            if self.fixed is False:
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv
            else:
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                    x = x.view(x.size(0), -1)  # flatten output of conv
            # print(x.shape)
            # input()
            # x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            # x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq
    

class ResCNNEncoder_gau(nn.Module):
    def __init__(self, cfg, fc_hidden1=512, fc_hidden2=512, drop_p=0.1, CNN_embed_dim=300):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(ResCNNEncoder_gau, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.fixed = cfg.MODEL.FIXED
        
        resnet = Resnet_gau_TFE(resnet='resnet34')
        # resnet = Resnet_gau(resnet='resnet18')

        # resnet = resnet34(pretrained=False)
        # resnet = resnet18(pretrained=False)
        # fc_features = resnet.fc.in_features
        # resnet.fc = torch.nn.Linear(fc_features, 3)
        if cfg.MODEL.WEIGHT_PRETRAIN != "":
            ckpt = torch.load(cfg.MODEL.WEIGHT_PRETRAIN, "cpu")
            resnet.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
                                   strict=True)
            if self.fixed is True:
                for p in resnet.parameters():
                    p.requires_grad = False
        # logging.info(list(resnet.children()))
        # input()
        # logging.info(list(resnet.children())[-2])
        # input()
        # logging.info(list(resnet.children())[-3])
        # input()
        
        # modules = list(resnet.children())[:-1]  # delete the last fc layer.
        # self.resnet = nn.Sequential(*modules)
        self.resnet = resnet
        # logging.info(self.resnet)
        self.fc1 = nn.Linear(512, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01, )
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01, )
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        # print(x_3d.size())
        for t in range(x_3d.size(1)):
            # ResNet CNN
            # print('x_ed:', type(x_3d))
            # print(x_3d.shape)
            # input()
            if self.fixed is False:
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv
            else:
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                    # print('after_resnet:', x.shape)
                    x = x.view(x.size(0), -1)  # flatten output of conv
                    # print('after_flatten:', x.shape)
            # print(x.shape)
            # input()
            # x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            # x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            # print(x.shape)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # print('cnn:', cnn_embed_seq.shape)
        return cnn_embed_seq


class DenseCNNEncoder(nn.Module):
    def __init__(self, cfg, fc_hidden1=512, fc_hidden2=512, drop_p=0.1, CNN_embed_dim=300):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(DenseCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.fixed = cfg.MODEL.FIXED

        densenet = densenet121(pretrained=False)
        fc_features = densenet.classifier.in_features
        densenet.classifier = torch.nn.Linear(fc_features, 3)
        if cfg.MODEL.WEIGHT_PRETRAIN != "":
            ckpt = torch.load(cfg.MODEL.WEIGHT_PRETRAIN, "cpu")
            densenet.load_state_dict({name: value for name, value in ckpt.pop('state_dict').items()},
                                   strict=True)
            if self.fixed is True:
                for p in densenet.parameters():
                    p.requires_grad = False
        # print(densenet)
        # modules = list(densenet.children())[:-1]  # delete the last fc layer.
        modules = densenet
        modules.classifier = DoNothing()
        self.densenet = modules
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', modules)
        # self.densenet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(fc_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01, )
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01, )
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        # print(x_3d.size())
        for t in range(x_3d.size(1)):
            # ResNet CNN
            if self.fixed is False:
                x = self.densenet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv
            else:
                with torch.no_grad():
                    x = self.densenet(x_3d[:, t, :, :, :])  # ResNet
                    x = x.view(x.size(0), -1)  # flatten output of conv

            # x = self.densenet(x_3d[:, t, :, :, :])  # ResNet
            # x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            # print(x.size())
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.1, num_classes=3):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class DecoderRNN_bd(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=512, h_FC_dim=256, drop_p=0.1, num_classes=3):
        super(DecoderRNN_bd, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN // 2,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class DecoderTFE(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_FC_dim=128, drop_p=0.1, num_classes=3, num=5):
        super(DecoderTFE, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.num = num

        self.TF = TransformerEncoderLayer(d_model=self.RNN_input_size, nhead=10)

        self.fc1 = nn.Linear(self.RNN_input_size * self.num, self.RNN_input_size)
        self.fc2 = nn.Linear(self.RNN_input_size, self.num_classes)

    def forward(self, x_RNN):
        # self.TF.flatten_parameters()
        # print(x_RNN.size())
        TF_out = self.TF(x_RNN)
        # print(TF_out.size())
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = TF_out.view(TF_out.size(0), -1)
        # print(x.size())
        x = self.fc1(x)  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class ResNet_LSTM(nn.Module):
    def __init__(self, cfg, num_classes=3, pretrained=False):
        super(ResNet_LSTM, self).__init__()
        self.encoder = ResCNNEncoder(cfg)
        self.decoder = DecoderRNN(num_classes=num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResNet_LSTM_bd(nn.Module):
    def __init__(self, cfg, num_classes=3, pretrained=False):
        super(ResNet_LSTM_bd, self).__init__()
        self.encoder = ResCNNEncoder(cfg)
        self.decoder = DecoderRNN_bd(num_classes=num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResNet_TFE(nn.Module):
    def __init__(self, cfg, num_classes=3, pretrained=False):
        super(ResNet_TFE, self).__init__()
        self.encoder = ResCNNEncoder(cfg)
        self.decoder = DecoderTFE(num_classes=num_classes, num=cfg.DATA.NUM)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DenseNet_LSTM_bd(nn.Module):
    def __init__(self, cfg, num_classes=3, pretrained=False):
        super(DenseNet_LSTM_bd, self).__init__()
        self.encoder = DenseCNNEncoder(cfg)
        self.decoder = DecoderRNN_bd(num_classes=num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DenseNet_TFE(nn.Module):
    def __init__(self, cfg, num_classes=3, pretrained=False):
        super(DenseNet_TFE, self).__init__()
        self.encoder = DenseCNNEncoder(cfg)
        self.decoder = DecoderTFE(num_classes=num_classes, num=cfg.DATA.NUM)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    
class ResNet_TFE_gau(nn.Module):
    def __init__(self, cfg, num_classes=3, pretrained=False):
        super(ResNet_TFE_gau, self).__init__()
        # self.encoder = ResCNNEncoder(cfg)
        self.encoder = ResCNNEncoder_gau(cfg)
        self.decoder = DecoderTFE(num_classes=num_classes, num=cfg.DATA.NUM)

    def forward(self, x):
        return self.decoder(self.encoder(x))