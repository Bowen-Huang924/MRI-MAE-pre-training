import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from lib.model.trans_resunet.transformer_module import TransformerLayers

__all__ = ['TransResUnet18V2']




class Finetune_TransResUnet18V2(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = ResEncoder()
        self.transformer_layers_16 = TransformerLayers(hide_dim=256, seq_len=16 * 16, layer_num=2, att_head_num=8)
        # self.decoder = Decoder(num_classes)
        # self.cls = ClsHead(2)

    def forward(self, x):
        x4, x8, x16 = self.encoder(x)

        b, c, h, w = x16.shape
        x16 = x16.flatten(2)
        x16 = x16.transpose(-1, -2)
        x16 = self.transformer_layers_16(x16)  # shape of x4 is [b, h*w, c]
        x16 = x16.permute(0, 2, 1)  # shape of x4 is [b, c, h*w]
        x16 = x16.contiguous().view(b, c, h, w)
        #
        # cls = self.cls(x16)
        # seg = self.decoder([x4, x8, x16])
        # print(seg.shape)

        return x4, x8, x16

class TransResUnet18V2_heatmap(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = ResEncoder()
        self.transformer_layers_16 = TransformerLayers(hide_dim=256, seq_len=16 * 16, layer_num=2, att_head_num=8)
        self.decoder = Decoder_heatmap(num_classes)
        self.cls = ClsHead(2)

    def forward(self, x):
        x4, x8, x16 = self.encoder(x)

        b, c, h, w = x16.shape
        x16 = x16.flatten(2)
        x16 = x16.transpose(-1, -2)
        x16 = self.transformer_layers_16(x16)  # shape of x4 is [b, h*w, c]
        x16 = x16.permute(0, 2, 1)  # shape of x4 is [b, c, h*w]
        x16 = x16.contiguous().view(b, c, h, w)

        cls = self.cls(x16)
        seg,x4 = self.decoder([x4, x8, x16])
        # print(seg.shape)
        # return seg, cls
        return seg, cls,x4, x8, x16


class TransResUnet18V2(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = ResEncoder()
        self.transformer_layers_16 = TransformerLayers(hide_dim=256, seq_len=16 * 16, layer_num=2, att_head_num=8)
        self.decoder = Decoder(num_classes)
        self.cls = ClsHead(2)

    def forward(self, x):
        x4, x8, x16 = self.encoder(x)

        b, c, h, w = x16.shape
        x16 = x16.flatten(2)
        x16 = x16.transpose(-1, -2)
        x16 = self.transformer_layers_16(x16)  # shape of x4 is [b, h*w, c]
        x16 = x16.permute(0, 2, 1)  # shape of x4 is [b, c, h*w]
        x16 = x16.contiguous().view(b, c, h, w)

        cls = self.cls(x16)
        seg = self.decoder([x4, x8, x16])
        # print(seg.shape)
        # return seg, cls
        return seg, cls


class TransResUnet18V2_cls(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = ResEncoder()
        self.transformer_layers_16 = TransformerLayers(hide_dim=256, seq_len=16 * 16, layer_num=2, att_head_num=8)
        # self.decoder = Decoder(num_classes)
        self.cls = ClsHead(2)

    def forward(self, x):
        x4, x8, x16 = self.encoder(x)

        b, c, h, w = x16.shape
        x16 = x16.flatten(2)
        x16 = x16.transpose(-1, -2)
        x16 = self.transformer_layers_16(x16)  # shape of x4 is [b, h*w, c]
        x16 = x16.permute(0, 2, 1)  # shape of x4 is [b, c, h*w]
        x16 = x16.contiguous().view(b, c, h, w)

        cls = self.cls(x16)
        # print(seg.shape)
        return cls
        # return seg, cls,x4, x8, x16



class TransResUnet18V2_tsne(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = ResEncoder()
        self.transformer_layers_16 = TransformerLayers(hide_dim=256, seq_len=16 * 16, layer_num=2, att_head_num=8)
        self.decoder = Decoder(num_classes)
        self.cls = ClsHead_tsne(2)

    def forward(self, x):
        x4, x8, x16 = self.encoder(x)

        b, c, h, w = x16.shape
        x16 = x16.flatten(2)
        x16 = x16.transpose(-1, -2)
        x16 = self.transformer_layers_16(x16)  # shape of x4 is [b, h*w, c]
        x16 = x16.permute(0, 2, 1)  # shape of x4 is [b, c, h*w]
        x16 = x16.contiguous().view(b, c, h, w)

        cls,y1 = self.cls(x16)
        seg = self.decoder([x4, x8, x16])
        # print(seg.shape)
        # return seg, cls
        return seg, cls,y1



class ResEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        res18 = resnet18()
        self.layers = nn.ModuleList()
        self.init_layer = nn.Sequential(
            res18.conv1,
            res18.bn1,
            res18.relu,
            res18.maxpool
        )
        self.layer1 = nn.Sequential(res18.layer1, copy.deepcopy(res18.layer1))
        self.layer2 = res18.layer2
        self.layer3 = res18.layer3
        self.transformer_layers_8 = TransformerLayers(hide_dim=128, seq_len=32 * 32, layer_num=2, att_head_num=8)

    def forward(self, x):
        x2 = self.init_layer(x)
        x4 = self.layer1(x2)  # x2 -> 1/4
        x8 = self.layer2(x4)  # x3 -> 1/8

        b, c, h, w = x8.shape
        x8 = x8.flatten(2)
        x8 = x8.transpose(-1, -2)
        x8 = self.transformer_layers_8(x8)  # shape of x4 is [b, h*w, c]
        x8 = x8.permute(0, 2, 1)  # shape of x4 is [b, c, h*w]

        x8 = x8.contiguous().view(b, c, h, w)
        x16 = self.layer3(x8)  # x4 -> 1/16

        return [x4, x8, x16]


class ClsHead(nn.Module):
    def __init__(self, cls_num):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 32, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 8, 3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)
        self.cls = nn.Linear(2048, cls_num)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        y = self.relu2(y)
        y = torch.flatten(y, 1)
        y = self.cls(y)
        return y


class ClsHead_tsne(nn.Module):
    def __init__(self, cls_num):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 32, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 8, 3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)
        self.cls = nn.Linear(2048, cls_num)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        y = self.relu2(y)
        y1 = torch.flatten(y, 1)
        # print('1',y1.shape)
        y = self.cls(y1)
        return y,y1

class Decoder_heatmap (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.up1 = Upsample(256, 128)
        self.conv1 = ResBlock(256, 256)
        self.up2 = Upsample(256, 64)
        self.conv2 = ResBlock(128, 128)
        self.logistic1 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(32, eps=1e-03)
        self.logistic2 = nn.ConvTranspose2d(32, num_classes, 3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        x4, x8, x16 = x
        x8 = torch.cat([self.up1(x16), x8], dim=1)
        x8 = self.conv1(x8)
        x4 = torch.cat([self.up2(x8), x4], dim=1)
        x4 = self.conv2(x4)
        y = self.logistic1(x4)
        y = self.bn(y)
        y = F.leaky_relu(y, 0.01)
        y = self.logistic2(y)

        return y,x4

class channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention, self).__init__()

        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x)
        max_pool_out = self.max_pooling(x).view([b, c])
        avg_pool_out = self.avg_pooling(x).view([b, c])
        # print(max_pool_out)

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out*x

class Cotraining_Decoder_ClsHead(nn.Module):
    def __init__(self, num_classes,cls_num):
        super().__init__()
        # seg---------------
        self.layers = nn.ModuleList()
        self.up1 = Upsample(256, 128)
        self.conv1 = ResBlock(256, 256)
        self.up2 = Upsample(256, 64)
        self.conv2 = ResBlock(128, 128)
        self.logistic1 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(32, eps=1e-03)
        self.logistic2 = nn.ConvTranspose2d(32, num_classes, 3, stride=2, padding=1, output_padding=1, bias=True)

        # cls----------------
        self.channel_attention1 = channel_attention(256)
        self.channel_attention2 = channel_attention(256)
        self.channel_attention3 = channel_attention(128)
        self.cls_conv1 = nn.Conv2d(256, 32, 3, stride=1, padding=1, bias=True)
        self.cls_conv1_2 = nn.Conv2d(256, 32, 3, stride=2, padding=1, bias=True)
        self.cls_conv1_3 = nn.Conv2d(128, 32, 3, stride=2, padding=1, bias=True)
        self.cls_conv1_4 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=True)
        self.cls_bn1 = nn.BatchNorm2d(96)
        self.cls_relu1 = nn.ReLU(inplace=True)
        self.cls_conv2 = nn.Conv2d(96, 8, 3, stride=1, padding=1, bias=True)
        self.cls_bn2 = nn.BatchNorm2d(8)
        self.cls_drop = nn.Dropout(0.5)
        self.cls_relu2 = nn.ReLU(inplace=True)
        self.cls_cls = nn.Linear(2048, cls_num)

    def forward(self, x):
        x4, x8, x16 = x
        att_x16 = self.channel_attention1(x16)
        # print('att_x16',att_x16.shape)
        x8 = torch.cat([self.up1(x16), x8], dim=1)
        att_x8 = self.channel_attention2(x8)
        # channel_attention2
        # print('att_x8',att_x8.shape)
        x8 = self.conv1(x8)
        x4 = torch.cat([self.up2(x8), x4], dim=1)
        att_x4 = self.channel_attention3(x4)
        # print('att_x4',att_x4.shape)
        x4 = self.conv2(x4)
        y = self.logistic1(x4)
        y = self.bn(y)
        y = F.leaky_relu(y, 0.01)
        seg_y = self.logistic2(y)
        # cls------------------
        y16 = self.cls_conv1(att_x16)
        y8 = self.cls_conv1_2(att_x8)
        y4 = self.cls_conv1_3(att_x4)
        y4 = self.cls_conv1_4(y4)
        # print('y16', y16.shape)
        # print('y8',y8.shape)
        # print('y4', y4.shape)
        cls_y = torch.cat([y16,y8,y4], dim=1)
        # print('cls_y', cls_y.shape)
        cls_y = self.cls_bn1(cls_y)
        cls_y = self.cls_relu1(cls_y)
        cls_y = self.cls_conv2(cls_y)
        cls_y = self.cls_bn2(cls_y)
        cls_y = self.cls_drop(cls_y)
        cls_y = self.cls_relu2(cls_y)
        # print('cls_y', cls_y.shape)
        cls_y = torch.flatten(cls_y, 1)
        cls_y = self.cls_cls(cls_y)
        return seg_y,cls_y

class Cotraining_Decoder_ClsHead_tsne(nn.Module):
    def __init__(self, num_classes,cls_num):
        super().__init__()
        # seg---------------
        self.layers = nn.ModuleList()
        self.up1 = Upsample(256, 128)
        self.conv1 = ResBlock(256, 256)
        self.up2 = Upsample(256, 64)
        self.conv2 = ResBlock(128, 128)
        self.logistic1 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(32, eps=1e-03)
        self.logistic2 = nn.ConvTranspose2d(32, num_classes, 3, stride=2, padding=1, output_padding=1, bias=True)

        # cls----------------
        self.channel_attention1 = channel_attention(256)
        self.channel_attention2 = channel_attention(256)
        self.channel_attention3 = channel_attention(128)
        self.cls_conv1 = nn.Conv2d(256, 32, 3, stride=1, padding=1, bias=True)
        self.cls_conv1_2 = nn.Conv2d(256, 32, 3, stride=2, padding=1, bias=True)
        self.cls_conv1_3 = nn.Conv2d(128, 32, 3, stride=2, padding=1, bias=True)
        self.cls_conv1_4 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=True)
        self.cls_bn1 = nn.BatchNorm2d(96)
        self.cls_relu1 = nn.ReLU(inplace=True)
        self.cls_conv2 = nn.Conv2d(96, 8, 3, stride=1, padding=1, bias=True)
        self.cls_bn2 = nn.BatchNorm2d(8)
        self.cls_drop = nn.Dropout(0.5)
        self.cls_relu2 = nn.ReLU(inplace=True)
        self.cls_cls = nn.Linear(2048, cls_num)

    def forward(self, x):
        x4, x8, x16 = x
        att_x16 = self.channel_attention1(x16)
        # print('att_x16',att_x16.shape)
        x8 = torch.cat([self.up1(x16), x8], dim=1)
        att_x8 = self.channel_attention2(x8)
        # channel_attention2
        # print('att_x8',att_x8.shape)
        x8 = self.conv1(x8)
        x4 = torch.cat([self.up2(x8), x4], dim=1)
        att_x4 = self.channel_attention3(x4)
        # print('att_x4',att_x4.shape)
        x4 = self.conv2(x4)
        y = self.logistic1(x4)
        y = self.bn(y)
        y = F.leaky_relu(y, 0.01)
        seg_y = self.logistic2(y)
        # cls------------------
        y16 = self.cls_conv1(att_x16)
        y8 = self.cls_conv1_2(att_x8)
        y4 = self.cls_conv1_3(att_x4)
        y4 = self.cls_conv1_4(y4)
        # print('y16', y16.shape)
        # print('y8',y8.shape)
        # print('y4', y4.shape)
        cls_y = torch.cat([y16,y8,y4], dim=1)
        # print('cls_y', cls_y.shape)
        cls_y = self.cls_bn1(cls_y)
        cls_y = self.cls_relu1(cls_y)
        cls_y = self.cls_conv2(cls_y)
        cls_y = self.cls_bn2(cls_y)
        cls_y = self.cls_drop(cls_y)
        cls_y = self.cls_relu2(cls_y)
        # print('cls_y', cls_y.shape)
        y1 = torch.flatten(cls_y, 1)
        cls_y = self.cls_cls(y1)
        return seg_y,cls_y,y1


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.up1 = Upsample(256, 128)
        self.conv1 = ResBlock(256, 256)
        self.up2 = Upsample(256, 64)
        self.conv2 = ResBlock(128, 128)
        self.logistic1 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(32, eps=1e-03)
        self.logistic2 = nn.ConvTranspose2d(32, num_classes, 3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        x4, x8, x16 = x
        x8 = torch.cat([self.up1(x16), x8], dim=1)
        x8 = self.conv1(x8)
        x4 = torch.cat([self.up2(x8), x4], dim=1)
        x4 = self.conv2(x4)
        y = self.logistic1(x4)
        y = self.bn(y)
        y = F.leaky_relu(y, 0.01)
        y = self.logistic2(y)

        return y




class ResBlock (nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_in // 2, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c_in // 2, c_out, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(c_in // 2, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(c_out, eps=1e-03)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.leaky_relu(y, 0.01)
        y = self.conv2(y)
        y = self.bn2(y)
        return F.leaky_relu(y + x, 0.01)


class Upsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch_in, ch_out, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(ch_out, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.01)
        return x


class Downsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out - ch_in, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(ch_out, eps=1e-3)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


if __name__ == '__main__':
    import time
    net = TransResUnet18V2()
    net.cuda()
    inp = torch.ones((1, 3, 256, 256)).cuda()
    pred_times = 200

    with torch.no_grad():
        for i in range(10):
            res = net(inp)

        total_t0 = time.time()
        for i in range(pred_times):
            print("\rpredicting {:d} / {:d}".format(i + 1, pred_times), end='')
            res = net(inp)
        print(res[1].shape)
        total_time = time.time() - total_t0

    print(total_time / pred_times)
    print(pred_times / total_time)
