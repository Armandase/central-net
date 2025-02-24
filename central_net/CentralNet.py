import torch
import torch.nn as nn
from decoders.MLP import MLP
from encoders.ResNet import BasicBlock, ExpansionResBlock


class CentralResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        nb_channels_mod1,
        nb_channels_mod2,
        num_classes,
        classifier=None,
        expansion=4,
        block=ExpansionResBlock,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.expansion = expansion
        self.in_channels = 64  # starting input size fo residual blocks
        # This value will increase by the factor of 4

        self.conv1_mod1 = nn.Conv2d(
            nb_channels_mod1,
            self.in_channels,
            kernel_size=7,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1_mod2 = nn.Conv2d(
            nb_channels_mod2,
            self.in_channels,
            kernel_size=7,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1_mod1 = nn.BatchNorm2d(self.in_channels)
        self.bn1_mod2 = nn.BatchNorm2d(self.in_channels)
        if block == BasicBlock:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()

        self.layer1_mod1 = self._layer(
            block, num_blocks[0], stride=1, inter_channels=64
        )
        self.layer2_mod1 = self._layer(
            block, num_blocks[1], stride=2, inter_channels=128
        )
        self.layer3_mod1 = self._layer(
            block, num_blocks[2], stride=2, inter_channels=256
        )
        self.layer4_mod1 = self._layer(
            block, num_blocks[3], stride=2, inter_channels=512
        )
        # reset in_channels to 64
        self.in_channels = 64

        self.layer1_mod2 = self._layer(
            block, num_blocks[0], stride=1, inter_channels=64
        )
        self.layer2_mod2 = self._layer(
            block, num_blocks[1], stride=2, inter_channels=128
        )
        self.layer3_mod2 = self._layer(
            block, num_blocks[2], stride=2, inter_channels=256
        )
        self.layer4_mod2 = self._layer(
            block, num_blocks[3], stride=2, inter_channels=512
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_mod1 = classifier
        self.classifier_mod2 = classifier

        self.relu = nn.ReLU()

    def forward(self, x_mod1, x_mod2):
        out_mod1 = self.relu(self.maxpool(self.bn1_mod1(self.conv1_mod1(x_mod1))))
        out_mod2 = self.relu(self.maxpool(self.bn1_mod2(self.conv1_mod2(x_mod2))))

        out_mod1 = self.layer1_mod1(out_mod1)
        out_mod1 = self.layer2_mod1(out_mod1)
        out_mod1 = self.layer3_mod1(out_mod1)
        out_mod1 = self.layer4_mod1(out_mod1)
        out_mod2 = self.layer1_mod2(out_mod2)
        out_mod2 = self.layer2_mod2(out_mod2)
        out_mod2 = self.layer3_mod2(out_mod2)
        out_mod2 = self.layer4_mod2(out_mod2)

        out_mod1 = self.avg_pool(out_mod1)
        out_mod2 = self.avg_pool(out_mod2)
        out_mod1 = out_mod1.view(out_mod1.shape[0], -1)
        out_mod2 = out_mod2.view(out_mod2.shape[0], -1)

        if self.classifier_mod1 is not None and self.classifier_mod2 is not None:
            out_mod1 = self.classifier_mod1(out_mod1)
            out_mod2 = self.classifier_mod2(out_mod2)

        return out_mod1, out_mod2

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


def get_central_net(num_classes=6, channels_mod1=3, channels_mod2=1):
    mlp = MLP(
        input_dim=512 * 4,
        output_dim=num_classes,
        dropout_rate=0.3,
    )
    model = CentralResNet(
        num_blocks=[3, 4, 6, 3],
        nb_channels_mod1=channels_mod1,
        nb_channels_mod2=channels_mod2,
        num_classes=num_classes,
        expansion=4,
        dropout_rate=0.0,
        classifier=mlp,
    )

    return model


if __name__ == "__main__":
    nb_channel_mod1 = 3
    nb_channel_mod2 = 1
    nb_classes = 6
    central = get_central_net(
        nb_classes, channels_mod1=nb_channel_mod1, channels_mod2=nb_channel_mod2
    )

    x_mod1 = torch.randn(1, nb_channel_mod1, 224, 224)
    x_mod2 = torch.randn(1, nb_channel_mod2, 224, 224)

    out_mod1, out_mod2 = central(x_mod1, x_mod2)
    print(out_mod1.shape, out_mod2.shape)
