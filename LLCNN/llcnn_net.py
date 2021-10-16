import torch
import torch.nn as nn
import torch.nn.functional as F


class CM(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        # 第一阶段的1x1卷积模块
        self.conv_1 = nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=True)
        # 第一阶段的两个3x3卷积模块
        self.conv_3_1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True)
        self.conv_3_2 = nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True)

        # 第二阶段的两个3x3卷积模块
        self.res_conv_1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True)
        self.res_conv_2 = nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True)

    def forward(self, x):
        # 第一阶段的多尺度聚合
        x_1 = self.conv_1(x)
        x_3_1 = F.relu(self.conv_3_1(x))
        x_3_2 = self.conv_3_2(x_3_1)
        x = x_3_2 + x_1

        # 第二阶段的残差连接
        x_res = F.relu(x)
        x_res = F.relu(self.res_conv_1(x_res))
        x_res = self.res_conv_2(x_res)
        out = F.relu(x_res + x)

        return out


class LLCNN(nn.Module):
    def __init__(self, num_cm=6, num_features=32):
        super().__init__()
        self.ncm = num_cm
        self.nf = num_features

        self.layers = []
        # 输入层
        self.layers.append(nn.Conv2d(3, self.nf, 3, 1, 1))
        self.layers.append(nn.ReLU())

        # 中间层，由多个CM模块组成
        for i in range(self.ncm):
            self.layers.append(CM(self.nf, self.nf))

        # 输出层
        self.layers.append(nn.Conv2d(self.nf, 3, 1, 1))

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    x = torch.rand(1, 3, 64, 64)
    llcnn = LLCNN(6, 32)
    print(llcnn(x).size())