import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F


class BaseRefineNet4Cascade(nn.Module):
    """Multi-path 4-Cascaded RefineNet for image segmentation

    Args:
        input_shape ((int, int, int)): (channel, h, w) assumes input has
            equal height and width
        refinenet_block (block): RefineNet Block
        num_classes (int, optional): number of classes
        features (int, optional): number of features in net
        resnet_factory (func, optional): A Resnet model from torchvision.
            Default: models.resnet101
        pretrained (bool, optional): Use pretrained version of resnet
            Default: True
        freeze_resnet (bool, optional): Freeze resnet model
            Default: True

    Raises:
        ValueError: size of input_shape not divisible by 32
    """
    def __init__(self):

        super().__init__()

        self.layer1 = nn.Conv3d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = nn.Conv3d(
            16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = nn.Conv3d(
            32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4 = nn.Conv3d(
            16, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        return layer_4 + x

