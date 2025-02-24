import torch
import torch.nn as nn
from encoders.ResNet import Bottleneck
from decoders.MLP import MLP


class CentralResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes,
        dropout_rate=0.0,
        nb_channel_mod1=3,
        nb_channel_mod2=1,
        classifier=None,
        fusion_method=None,
    ):
        super(CentralResNet, self).__init__()
        self.in_channels = 64

        # resnet stem mod1 part
        self.conv1_mod1 = nn.Conv2d(
            nb_channel_mod1,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.bn1_mod1 = nn.BatchNorm2d(self.in_channels)

        # resnet stem mod2 part
        self.conv1_mod2 = nn.Conv2d(
            nb_channel_mod2,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.bn1_mod2 = nn.BatchNorm2d(self.in_channels)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # res blocks modality 1
        self.layer1_mod1 = self._make_layer(
            block=block, blocks=layers[0], out_channels=64, stride=1
        )
        self.layer2_mod1 = self._make_layer(
            block=block, blocks=layers[1], out_channels=128, stride=2
        )
        self.layer3_mod1 = self._make_layer(
            block=block, blocks=layers[2], out_channels=256, stride=2
        )
        self.layer4_mod1 = self._make_layer(
            block=block, blocks=layers[3], out_channels=512, stride=2
        )

        # res blocks modality 2
        self.layer1_mod2 = self._make_layer(
            block=block, blocks=layers[0], out_channels=64, stride=1
        )
        self.layer2_mod2 = self._make_layer(
            block=block, blocks=layers[1], out_channels=128, stride=2
        )
        self.layer3_mod2 = self._make_layer(
            block=block, blocks=layers[2], out_channels=256, stride=2
        )
        self.layer4_mod2 = self._make_layer(
            block=block, blocks=layers[3], out_channels=512, stride=2
        )

        # classifier block
        self.adappool = nn.AdaptiveAvgPool2d((2, 2))
        if classifier is None:
            self.classifier_mod1 = nn.Linear(512 * block.expansion, num_classes)
            self.classifier_mod2 = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.classifier_mod1 = classifier(512 * block.expansion, num_classes)
            self.classifier_mod2 = classifier(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):

        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(num_features=out_channels * block.expansion),
            )

        layers = []

        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x_mod1, x_mod2):

        x_mod1 = self.relu(self.bn1_mod1(self.conv1_mod1(x_mod1)))
        x_mod1 = self.maxpool(x_mod1)
        x_mod1 = self.layer1_mod1(x_mod1)
        x_mod1 = self.layer2_mod1(x_mod1)
        x_mod1 = self.layer3_mod1(x_mod1)
        x_mod1 = self.layer4_mod1(x_mod1)
        x_mod1 = self.adppool(x_mod1)
        x_mod1 = torch.flatten(x_mod1, 1)

        x_mod2 = self.relu(self.bn1_mod2(self.conv1_mod2(x_mod2)))
        x_mod2 = self.maxpool(x_mod2)
        x_mod2 = self.layer1_mod2(x_mod2)
        x_mod2 = self.layer2_mod2(x_mod2)
        x_mod2 = self.layer3_mod2(x_mod2)
        x_mod2 = self.layer4_mod2(x_mod2)
        x_mod2 = self.adppool(x_mod2)
        x_mod2 = torch.flatten(x_mod2, 1)

        return self.classifier_mod1(x_mod1), self.classifier_mod2(x_mod2)


def get_central_net(num_classes=6, channels_rgb=3, channels_nir=22):
    model = CentralResNet(
        Block,
        [3, 4, 6, 3],
        num_classes,
        dropout_rate=0.0,
        nb_channel_mod1=channels_rgb,
        nb_channel_mod2=channels_nir,
        classifier=MLP,
    )
    return model


if __name__ == "__main__":
    central = get_central_net()

    x_mod1 = torch.randn(1, 3, 224, 224)
    x_mod2 = torch.randn(1, 22, 224, 224)

    out_mod1, out_mod2 = central(x_mod1, x_mod2)
