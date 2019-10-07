import torch
from torch import nn
from . import model_helpers


class ParametrizedConcatPool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.apool = nn.AvgPool2d(kernel_size)
        self.mpool = nn.MaxPool2d(kernel_size)

        self.ascale = nn.Parameter(torch.ones(1))
        self.ashift = nn.Parameter(torch.zeros(1))

        self.mscale = nn.Parameter(torch.ones(1))
        self.mshift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.cat([
                self.ascale * self.apool(x) + self.ashift,
                self.mscale * self.mpool(x) + self.mshift
            ], dim=1)


class ConcatPool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.apool = nn.AvgPool2d(kernel_size)
        self.mpool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        return torch.cat([self.apool(x), self.mpool(x)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, inf, outf, expand, neg_slope, downsample=False):
        super().__init__()

        # Expand Block
        c = int(inf * expand)
        self.downsample = downsample
        if downsample:
            mid_stride = 2
        else:
            mid_stride = 1
        self.trunk = nn.Sequential(
            nn.Conv2d(inf, c, 1, stride=1, padding=0),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),

            nn.Conv2d(c, c, 3, stride=mid_stride, padding=1, groups=c),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),

            nn.Conv2d(c, outf, 1, stride=1, padding=0),
            nn.BatchNorm2d(outf)
        )

        # Initialize weights
        model_helpers.weight_init(self.trunk)
        # Initialize last batchnorm after the residual connnection (Xie et al)
        if not self.downsample:
            nn.init.zeros_(self.trunk[-1].weight)
            nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, x):
        y = self.trunk(x)
        # Skip connection based on whether or not downsampling occurred
        if self.downsample:
            return y
        return x + y


class ResBlock2(nn.Module):
    def __init__(self, inf, outf, expand, neg_slope, downsample=False, parametrized_pool=False):
        super().__init__()

        # Expand Block
        c = int(inf * expand)
        self.downsample = downsample
        if downsample:
            mid_stride = 2
        else:
            mid_stride = 1
        self.trunk = nn.Sequential(
            nn.Conv2d(inf, c, 1, stride=1, padding=0),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),

            nn.Conv2d(c, c, 3, stride=mid_stride, padding=1, groups=c),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),

            nn.Conv2d(c, outf, 1, stride=1, padding=0),
            nn.BatchNorm2d(outf)
        )

        # Initialize weights
        model_helpers.weight_init(self.trunk)
        # Initialize last batchnorm after the residual connnection (Xie et al)
        if downsample:
            if parametrized_pool:
                self.pool = ParametrizedConcatPool(2)
            else:
                self.pool = ConcatPool(2)

        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, x):
        y = self.trunk(x)
        # Skip connection based on whether or not downsampling occurred
        if self.downsample:
            return self.pool(x) + y
        return x + y


class ResBlock3(nn.Module):
    def __init__(self, inf, outf, expand, neg_slope, downsample=False):
        super().__init__()

        # Expand Block
        c = int(inf * expand)
        self.downsample = downsample
        if downsample:
            mid_stride = 2
        else:
            mid_stride = 1
        self.trunk = nn.Sequential(
            nn.Conv2d(inf, c, 1, stride=1, padding=0),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),

            nn.Conv2d(c, c, 3, stride=mid_stride, padding=1, groups=c),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),

            nn.Conv2d(c, outf, 1, stride=1, padding=0),
            nn.BatchNorm2d(outf)
        )

        # Initialize weights
        model_helpers.weight_init(self.trunk)
        # Initialize last batchnorm after the residual connnection (Xie et al)
        if downsample:
            self.pool = ParametrizedConcatPool(2)
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, x):
        y = self.trunk(x)
        # Skip connection based on whether or not downsampling occurred
        if self.downsample:
            return self.pool(x) + y
        return x + y


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, alpha=1, dropout=0.3, neg_slope=0.2, block=ResBlock):
        super().__init__()

        # Initial convolution for channel expansion
        self.init_conv = nn.Conv2d(3, 32 * alpha, 3, stride=1, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32 * alpha)
        self.lrelu = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

        # Initialize layers
        nn.init.kaiming_normal_(self.init_conv.weight)
        nn.init.ones_(self.init_bn.weight)
        nn.init.zeros_(self.init_bn.bias)

        self.features = nn.Sequential(
                    block(32 * alpha, 32, alpha, neg_slope),
                    block(32, 64, alpha, neg_slope, downsample=True),  # 16x16
                    block(64, 64, alpha, neg_slope),
                    block(64, 128, alpha, neg_slope, downsample=True),  # 8x8
                    block(128, 128, alpha, neg_slope),
                    block(128, 256, alpha, neg_slope, downsample=True),  # 4x4
                    block(256, 256, alpha, neg_slope),
                    block(256, 512, alpha, neg_slope, downsample=True)  # 2x2
                )

        self.pool = nn.AvgPool2d(2)

        self.classifier = nn.Sequential(
                nn.Dropout2d(dropout),
                nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.lrelu(x)

        # Feature extraction
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return self.classifier(x)