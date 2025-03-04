import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


class DoNothing(nn.Module):

    def forward(self, x):
        return x


class LSTMDenseNet(nn.Module):

    def __init__(self, output_size, pretrained=True, lock=True):
        super(LSTMDenseNet, self).__init__()
        self.densenet = torchvision.models.densenet121(pretrained=pretrained)
        self.num_features = self.densenet.classifier.in_features
        self.densenet.classifier = DoNothing()
        self.lstm = nn.LSTM(self.num_features, output_size * 2, 2, bidirectional=True)
        self.fc = nn.Linear(output_size * 2 * 5 * 2, 3)
        self.lock = lock

    def forward(self, frames):
        # print('frames', frames.size())
        batch_size = frames.size()[0]
        size_a = frames.size()[-2]
        size_b = frames.size()[-1]
        frames = frames.view(-1, 3, size_a, size_b)
        output = self.densenet(frames)
        if self.lock:
            output = output.clone().detach()
        # print('output', output.size())
        output = output.view(batch_size, -1, self.num_features)
        # print('output', output.size())
        ans, _ = self.lstm(output)
        ans = ans.view(batch_size, 3 * 2 * 5 * 2)
        ans = self.fc(ans)
        # print('ans', ans.size())

        return ans, output


class TransformerDenseNet(nn.Module):

    def __init__(self, output_size, pretrained=True, lock=False):
        super(TransformerDenseNet, self).__init__()
        self.densenet = torchvision.models.densenet121(pretrained=pretrained)
        self.num_features = self.densenet.classifier.in_features
        self.densenet.classifier = DoNothing()
        self.tf = TransformerEncoderLayer(d_model=1024, nhead=8, dropout=0.0)
        self.fc = nn.Linear(1024 * 5, output_size)
        self.lock = lock

    def forward(self, frames):
        # print('frames', frames.size())
        batch_size = frames.size()[0]
        size_a = frames.size()[-2]
        size_b = frames.size()[-1]
        frames = frames.view(-1, 3, size_a, size_b)
        output = self.densenet(frames)
        if self.lock:
            output = output.clone().detach()
        print('output', output.size())
        output = output.view(batch_size, -1, self.num_features)
        print('output', output.size())
        ans = self.tf(output)
        # ans = output
        print('ans', ans.size())
        ans = ans.view(batch_size, 1024 * 5)
        ans = self.fc(ans)
        # print('ans', ans.size())

        return ans