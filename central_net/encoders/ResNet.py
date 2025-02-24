"""
Implementation of the ResNet
ResNet paper by Kaiming He et al: 
https://arxiv.org/abs/1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ExpansionResBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, downsampling_layer=None, stride=1):
        super().__init__()
        expansion = 4

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=inter_channels * expansion,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(inter_channels * expansion)
        self.relu = nn.ReLU()
        self.downsampling_layer = downsampling_layer

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsampling_layer is not None:
            res = self.downsampling_layer(res)

        out += res
        out = F.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling_layer=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsampling_layer = downsampling_layer

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsampling_layer is not None:
            res = self.downsampling_layer(res)

        out += res
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        img_channels,
        num_classes,
        classifier=None,
        expansion=4,
        block=ExpansionResBlock,
    ):
        super().__init__()
        self.expansion = expansion
        self.in_channels = 64  # starting input size fo residual blocks
        # This value will increase by the factor of 4

        self.conv1 = nn.Conv2d(
            img_channels,
            self.in_channels,
            kernel_size=7,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        if block == BasicBlock:
            print("Using BasicBlock")
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()

        self.layer1 = self._layer(block, num_blocks[0], stride=1, inter_channels=64)
        self.layer2 = self._layer(block, num_blocks[1], stride=2, inter_channels=128)
        self.layer3 = self._layer(block, num_blocks[2], stride=2, inter_channels=256)
        self.layer4 = self._layer(block, num_blocks[3], stride=2, inter_channels=512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)

        if self.classifier is not None:
            out = self.classifier(out)

        return out

    def _layer(self, block, num_blocks, stride, inter_channels):
        layers = []
        downsampling_layer = None
        if stride != 1 or self.in_channels != inter_channels * self.expansion:
            downsampling_layer = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    inter_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(inter_channels * self.expansion),
            )

        layers.append(
            block(self.in_channels, inter_channels, downsampling_layer, stride)
        )
        self.in_channels = inter_channels * self.expansion

        for _ in range(num_blocks - 1):
            layers.append(block(self.in_channels, inter_channels))

        return nn.Sequential(*layers)


def ResNet18(img_channels, num_classes, classifier=None):
    model = ResNet(
        num_blocks=[2, 2, 2, 2],
        img_channels=img_channels,
        num_classes=num_classes if classifier is not None else None,
        classifier=classifier,
        expansion=1,
        block=BasicBlock,
    )
    return model


def ResNet32(img_channels, num_classes, classifier=None):
    model = ResNet(
        num_blocks=[3, 4, 6, 3],
        img_channels=img_channels,
        num_classes=num_classes if classifier is not None else None,
        classifier=classifier,
        expansion=1,
        block=BasicBlock,
    )
    return model


def ResNet50(img_channels, num_classes, classifier=None):
    model = ResNet(
        num_blocks=[3, 4, 6, 3],
        img_channels=img_channels,
        num_classes=num_classes if classifier is not None else None,
        classifier=classifier,
    )
    return model


def ResNet101(img_channels, num_classes, classifier=None):
    model = ResNet(
        num_blocks=[3, 4, 23, 3],
        img_channels=img_channels,
        num_classes=num_classes if classifier is not None else None,
        classifier=classifier,
    )
    return model


def ResNet152(img_channels, num_classes, classifier=None):
    model = ResNet(
        num_blocks=[3, 8, 36, 3],
        img_channels=img_channels,
        num_classes=num_classes if classifier is not None else None,
        classifier=classifier,
    )
    return model


if __name__ == "__main__":
    nb_class = 10
    expansion = 1

    MLP_50 = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, nb_class),
    )

    MLP_18 = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, nb_class),
    )
    # model = ResNet50(img_channels=3, num_classes=nb_class, classifier=MLP)
    model = ResNet50(img_channels=3, num_classes=nb_class, classifier=MLP_50)
    model = ResNet18(img_channels=3, num_classes=nb_class, classifier=MLP_18)

    model = model.to("cuda")
    summary(model, (3, 224, 224))
    dummy = torch.randn(1, 3, 224, 224).to("cuda")
    out = model(dummy)
    print(out.shape)
