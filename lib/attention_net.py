import torch
from torch import nn
from torch.nn import functional as F  # 调用F.函数
import os


class ResBlk(nn.Module):  # 定义Resnet Block模块
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=2):  # 进入网络前先得知道传入层数和传出层数的设定
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()  # 初始化

        # we add stride support for resbok, which is distinct from tutorials.
        # 根据resnet网络结构构建2个（block）块结构 第一层卷积 卷积核大小3*3,步长为1，边缘加1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # 将第一层卷积处理的信息通过BatchNorm2d
        self.bn1 = nn.BatchNorm2d(ch_out)
        # 第二块卷积接收第一块的输出，操作一样
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 确保输入维度等于输出维度
        self.extra = nn.Sequential()  # 先建一个空的extra
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):  # 定义局部向前传播函数
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))  # 对第一块卷积后的数据再经过relu操作
        out = self.bn2(self.conv2(out))  # 第二块卷积后的数据输出
        # print(x.shape)
        # print(out.shape)
        out = self.extra(x) + out  # 将x传入extra经过2块（block）输出后与原始值进行相加
        out = F.relu(out)  # 调用relu，这里使用F.调用

        return out


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


class spatial_attention(nn.Module):
    def __init__(self, kernel_size = 3):
        super(spatial_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, 7//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # print('max_pool.shape', max_pool_out.shape)
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out * x

class ResNet18(nn.Module):  # 构建resnet18层

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(  # 首先定义一个卷积层
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        # followed 4 blocks 调用4次resnet网络结构，输出都是输入的2倍
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBlk(16, 32, stride=2)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(32, 64, stride=2)
        # # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(64, 128, stride=2)
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(128, 256, stride=2)
        # self.attention = CBAM(channel=256)
        self.channel = channel_attention(channel=256)
        self.spatial = spatial_attention()
        self.OPM = nn.Linear(256, 3)
        # self.OPM2 = nn.Linear(512, 128)
        # self.OPM3 = nn.Linear(128, 31)

        # self.SNR1 = nn.Linear(1024+3, 512)
        self.SNR1 = nn.Linear(256 + 3, 64)
        self.SNR2 = nn.Linear(64, 1)
        self.turb_g1 = nn.Linear(256, 64)
        self.turb_g2 = nn.Linear(64, 1)
        self.turb_p = nn.Linear(256, 1)

    def forward(self, x):  # 定义整个向前传播
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))  # 先经过第一层卷积
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)  # 然后通过4次resnet网络结构
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = self.channel(x)
        x = self.spatial(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)  # 平铺一维值
        y = x
        # print('y.shape:', y.shape)
        opm = self.OPM(x)
        # print('x.shape:', x.shape)
        snr = F.relu(self.SNR1(torch.cat([opm, y], dim=1)))
        snr = self.SNR2(snr)
        turb_g = F.relu(self.turb_g1(x))
        turb_g = self.turb_g2(turb_g)

        turb_p = self.turb_p(x)

        return snr, opm, turb_g, turb_p


def print_size_of_model(model):
    torch.save(model.state_dict, "temp.mdl")
    print('size(MB):', os.path.getsize("temp.mdl") / 1e6)
    os.remove("temp.mdl")


def main():  # 测试代码
    device = torch.device('cpu')
    # device = torch.device('cuda')

    blk = ResBlk(64, 128, stride=4)  # 确定resnet block 的输入维度和输出维度
    tmp = torch.randn(2, 64, 224, 224)  # 输入数据
    out = blk(tmp)  # 调用resnet网络
    print('block:', out.shape)  # 打印结构
    x = torch.randn(2, 3, 224, 224)  # 输入图片信息 这里相当与2张32*32大小的彩图
    x.to(device)
    model = ResNet18().to(device)  # 调用resnet18整个网络结构
    out = model(x)
    print('out', out)
    # print(len(out))
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameter size:', p)
    # print('resnet_snr:', out[0].shape)
    # print('resnet_opm:', out[1].shape)
    # print('resnet_turb_g:', out[2].shape)
    # print('resnet_turb_p:', out[3].shape)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    print_size_of_model(model)
    print_size_of_model(quantized_model)
    # print('resnet_snr:', out[1].shape)


if __name__ == '__main__':
    main()