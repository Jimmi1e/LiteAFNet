import torch
import torch.nn as nn

""" SC_Squeeze and Excitation block """
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)
        q = self.norm(q)
        return U * q

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)
        z = self.Conv_Squeeze(z)
        z = self.Conv_Excitation(z)
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)



        self.se = scSE(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

class MixPool(nn.Module):
    def __init__(self, in_c, out_c):
        super(MixPool, self).__init__()

        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        fmask = (self.fmask(x) > 0.5).type(torch.cuda.FloatTensor)
        m = nn.MaxPool2d((m.shape[2]//x.shape[2], m.shape[3]//x.shape[3]))(m)
        x1 = x * torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], axis=1)
        return x