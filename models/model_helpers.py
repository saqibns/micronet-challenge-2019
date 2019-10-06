from torch import nn


def weight_init(trunk):
    for module in trunk:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)

        if isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)