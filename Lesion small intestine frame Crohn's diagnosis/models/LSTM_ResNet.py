import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import torch
from .resnet import *
from .densenet import *
from .vit import Transformer, repeat, rearrange
import math


class DoNothing(nn.Module):

    def forward(self, x):
        return x


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


class TFC_backup(nn.Module):

    def __init__(self, output_size, fc_features=512):
        super(TFC, self).__init__()
        self.fc_features = fc_features
        self.tf = TransformerEncoderLayer(d_model=self.fc_features, nhead=5)
        self.fc = nn.Linear(self.fc_features * 22, self.fc_features * 2)
        self.tf2 = TransformerEncoderLayer(d_model=self.fc_features * 2, nhead=2)
        self.fc2 = nn.Linear(self.fc_features * 2 * 22, output_size)
        self.twostage = True

    def forward(self, frames):
        batch_size, len_seq = frames.size()[0], frames.size()[1]

        if self.twostage:
            seq_mini = int(len_seq ** 0.5)
            # print(seq_mini)
            ans = None
            for i in range(seq_mini):
                output_mini = frames[:, i * seq_mini: (i + 1) * seq_mini, :]
                # print('----------output1', output_mini.size())
                output_mini = output_mini.view(batch_size, -1, self.fc_features)
                # print('----------output2', output_mini.size())
                ans_mini = self.tf(output_mini)
                # print('----------ans1', ans_mini.size())
                ans_mini = ans_mini.view(batch_size, self.fc_features * 22)
                ans_mini = self.fc(ans_mini)
                # print('----------ans2', ans_mini.size())
                if i == 0:
                    ans = ans_mini
                else:
                    ans = torch.cat((ans, ans_mini), 1)
                    # print('----------ans3', ans.size())

        else:
            output = frames
            # ans = output

            output = output.view(batch_size, -1, self.fc_features)
            # print('----------output', output.size())
            ans = self.tf(output)
            # print('----------ans1', ans.size())
            ans = ans.view(batch_size, self.fc_features * 1000)
            # print('----------ans2', ans.size())
            ans = self.fc(ans)
        # print('----------ans3', ans.size())
        ans = ans.view(batch_size, -1, self.fc_features * 2)
        # print('----------ans4', ans.size())
        ans = self.tf2(ans)
        # print('----------ans5', ans.size())

        ans = ans.view(batch_size, self.fc_features * 2 * 22)
        # ans = self.fc(ans)
        ans = self.fc2(ans)
        # print('ans', ans.size())

        return ans


class TFC(nn.Module):
    def __init__(self, num_classes, fc_features=513):
        super().__init__()

        self.num_patches = 498
        emb_dropout = 0.1
        dropout = 0.1
        pool = 'cls'
        dim = fc_features

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, fc_features))
        # self.pos_embedding = None

        self.dropout = nn.Dropout(emb_dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(fc_features, 6, 16, 64, 2048, dropout)

        self.transformer2 = Transformer(fc_features, 6, 16, 64, 2048, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(fc_features),
            nn.Linear(fc_features, num_classes)
        )

    def forward(self, x):
        x = x[:, :self.num_patches]

        # self.pos_embedding = x[:, :self.num_patches + 1, :1]
        # x = x[:, :self.num_patches, 1:]

        b, n, _ = x.shape
        # print('x1', x.size(), self.pos_embedding.size())
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('x2', x.size())
        x += self.pos_embedding[:, :(n + 1)]
        # print('x3', x.size())
        x = self.dropout(x)
        # print('x4', x.size())

        x = self.transformer(x)
        # print('x5', x.size())

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # print('x6', x.size())

        x = self.to_latent(x)
        # print('x7', x.size())
        x = self.mlp_head(x)
        # print('x8', x.size())
        return x


class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, position_embedding_type):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def forward_rope(self, x):
        """
        return (b l d)
        """
        embeddings = self.embeddings[None, :x.size(1), :]  # b l d
        embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
        sin, cos = embeddings.unbind(dim=-2)  # b l d//2
        sin, cos = map(lambda t: repeat(t, '... d -> ... (d 2)'), (sin, cos))  # b l d
        return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x):
        if self.position_embedding_type == 'fixed':
            return self.forward_fixed(x)

        elif self.position_embedding_type == 'rope':
            return self.forward_rope(x)

    def forward(self, x, _=None):
        if x.dim() == 3:
            return self._forward(x)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x


class GivenAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, position_embedding_type):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        self.inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        # print(position.size(), self.inv_freq.size())
        sinusoid_inp = torch.einsum('i,j -> ij', position, self.inv_freq)
        # print(sinusoid_inp.size())
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # print(embeddings.size())
        self.register_buffer('embeddings', embeddings)
        self.inv_freq = self.inv_freq.cuda()

    def forward(self, x, idx):
        b = idx.size(0)
        idx_tokens = torch.zeros((b, 1)).cuda()
        idx = torch.cat((idx_tokens, idx), dim=1)
        # print('input:', idx.size(), self.inv_freq.size(), type(idx), type(self.inv_freq), idx.device, self.inv_freq.device)
        sinusoid_inp = torch.einsum('bi,j -> bij', idx, self.inv_freq)
        sinusoid_inp = sinusoid_inp[:idx.size(1)]
        # print('sinusoid_inp', sinusoid_inp.size())
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # print('embeddings', embeddings.size(), self.embeddings[None, :x.size(1), :].size())
        # return x + self.embeddings[None, :x.size(1), :]
        return x + embeddings


class RelativePositionEmbedding(nn.Module):
    def __init__(self,
                 relative_attention_num_buckets, num_attention_heads,
                 hidden_size, position_embedding_type='bias'):

        super().__init__()

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.position_embedding_type = position_embedding_type
        self.num_attention_heads = num_attention_heads
        self.is_absolute = False

        if position_embedding_type == 'bias':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, num_attention_heads)

        elif position_embedding_type == 'contextual(1)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)
            self.to_r = nn.Linear(hidden_size, hidden_size, bias=False)

        elif position_embedding_type == 'contextual(2)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)

    def compute_bias(self, q, k, to_q=None, to_k=None):
        """
        q, k: [b h l d]
        return [b h l l]
        """
        h = self.num_attention_heads
        query_position = torch.arange(q.size(2), dtype=torch.long, device=self.embeddings.weight.device)[:, None]
        key_position = torch.arange(k.size(2), dtype=torch.long, device=self.embeddings.weight.device)[None, :]

        relative_position = query_position - key_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets
        )

        if self.position_embedding_type == 'bias':
            bias = self.embeddings(relative_position_bucket)
            bias = rearrange(bias, 'm n h -> 1 h m n')

        elif self.position_embedding_type == 'contextual(1)':
            r = self.embeddings(relative_position_bucket)
            r = self.to_r(r)
            r = rearrange(r, 'm n (h d) -> h m n d', h=h)

            bias = torch.einsum('b h m d, h m n d -> b h m n', q, r)

        elif self.position_embedding_type == 'contextual(2)':
            r = self.embeddings(relative_position_bucket)

            kr = to_k(r)
            qr = to_q(r)

            kr = rearrange(kr, 'm n (h d) -> h m n d', h=h)
            qr = rearrange(qr, 'm n (h d) -> h m n d', h=h)

            bias1 = torch.einsum('b h m d, h m n d -> b h m n', q, kr)
            bias2 = torch.einsum('b h n d, h m n d -> b h m n', k, qr)

            bias = bias1 + bias2

        return bias

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance=128):
        """
        relative_position: [m n]
        """

        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


class TF2(nn.Module):
    def __init__(self, num_classes, fc_features=512):
        super().__init__()

        self.num_patches = 450
        emb_dropout = 0.2
        dropout = 0.2
        pool = 'cls'
        dim = fc_features
        pnum = 7  # 4  # 32

        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, fc_features))
        self.pos_embedding = FixedAbsolutePositionEmbedding(self.num_patches + 1, dim, 'fixed')
        # self.pos_embedding = GivenAbsolutePositionEmbedding(self.num_patches + 1, dim, 'fixed')
        self.pos_embedding2 = FixedAbsolutePositionEmbedding(pnum + 1, dim, 'fixed')
        # self.pos_embedding2 = RelativePositionEmbedding(2, pnum + 1, dim)

        self.dropout = nn.Dropout(emb_dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(fc_features, 6, 16, 64, 2048, dropout)
        self.transformer2 = Transformer(fc_features, 2, 16, 64, 2048, dropout) # 2_2

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(fc_features),
            nn.Linear(fc_features, num_classes)
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(fc_features),
            nn.Linear(fc_features, num_classes)
        )

    def forward(self, x4, idx4):
        # x4 (batch_size, pnum, num_patches, fc_features) pnum = 4, num_patches = 450
        # idx4 (batch_size, pnum, num_patches)
        # output (batch_size, num_classes)
        preds = []
        tokens = []
        pnum = 4  # 32
        for i in range(pnum):
            x = x4[:, i, :self.num_patches]
            idx = idx4[:, i, :self.num_patches]

            # self.pos_embedding = x[:, :self.num_patches + 1, :1]
            # x = x[:, :self.num_patches, 1:]

            b, n, dim = x.shape
            # print('x1', x.size(), self.pos_embedding.size())
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            # print('x2', x.size())
            # TODO: pos_embedding
            # x += self.pos_embedding[:, :(n + 1)]
            x = self.pos_embedding(x, idx)
            # print('x3', x.size())
            x = self.dropout(x)
            # print('x4', x.size())

            x = self.transformer(x)
            # print('x5', x.size())

            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            # print('x6', x.size())
            tokens.append(x)

            x = self.to_latent(x)
            # print('x7', x.size())
            preds.append(self.mlp_head(x))
            # print('x8', x.size())
        tokens = torch.stack(tokens, 1)
        b, n, _ = tokens.shape
        cls_tokens = repeat(self.cls_token2, '() n d -> b n d', b=b)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.pos_embedding2(tokens)
        tokens = self.transformer2(tokens)
        tokens = tokens.mean(dim=1) if self.pool == 'mean' else tokens[:, 0]
        output = self.mlp_head2(tokens)

        return output


class TF1(nn.Module):
    def __init__(self, num_classes, fc_features=512):
        super().__init__()

        emb_dropout = 0.2
        dropout = 0.2
        pool = 'cls'
        dim = fc_features

        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, fc_features))
        # self.pos_embedding = FixedAbsolutePositionEmbedding(self.num_patches + 1, dim, 'fixed')
        self.pos_embedding = GivenAbsolutePositionEmbedding(1005, dim, 'fixed')
        # self.pos_embedding2 = FixedAbsolutePositionEmbedding(pnum + 1, dim, 'fixed')
        # self.pos_embedding2 = RelativePositionEmbedding(2, pnum + 1, dim)

        self.dropout = nn.Dropout(emb_dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(fc_features, 6, 16, 64, 2048, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(fc_features),
            nn.Linear(fc_features, num_classes)
        )

    def forward(self, x, idx):

        x = x[:, 0]
        idx = idx[:, 0]
        idx = (idx * 10) // 10
        b, n, dim = x.shape
        # print('x1', x.size(), self.pos_embedding.size())
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('x2', x.size())
        # TODO: pos_embedding
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.pos_embedding(x, idx)
        # print('x3', x.size())
        x = self.dropout(x)
        # print('x4', x.size())

        x = self.transformer(x)
        # print('x5', x.size())

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # print('x6', x.size())

        x = self.to_latent(x)
        # print('x7', x.size())
        # print('x8', x.size())
        output = self.mlp_head(x)

        return output


# 2D CNN encoder using ResNet-18 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, cfg, fc_hidden1=512, fc_hidden2=512, drop_p=0.1, CNN_embed_dim=300):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.fixed = cfg.MODEL.FIXED

        resnet = resnet34(pretrained=False)
        fc_features = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(fc_features, 2)
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
    def __init__(self, CNN_embed_dim=300, h_FC_dim=128, drop_p=0.1, num_classes=2, num=5):
        super(DecoderTFE, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.num = num

        self.TF = TransformerEncoderLayer(d_model=self.RNN_input_size, nhead=10)

        self.fc1 = nn.Linear(self.RNN_input_size * self.num, self.RNN_input_size)
        self.fc2 = nn.Linear(self.RNN_input_size, self.num_classes)

    def forward(self, x_RNN, output_feature=False):
        # self.TF.flatten_parameters()
        # print(x_RNN.size())
        TF_out = self.TF(x_RNN)
        # print(TF_out.size())
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = TF_out.view(TF_out.size(0), -1)
        # print(x.size())
        if output_feature:
            feature = x
        x = self.fc1(x)  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        if output_feature:
            return x, feature

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
    def __init__(self, cfg, num_classes=2, pretrained=False):
        super(DenseNet_TFE, self).__init__()
        self.encoder = DenseCNNEncoder(cfg)
        self.decoder = DecoderTFE(num_classes=num_classes, num=cfg.DATA.NUM)

    def forward(self, x):
        return self.decoder(self.encoder(x))
