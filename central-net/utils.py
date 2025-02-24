import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        dropout_prob=0.4,
        alpha_central=1,
        alpha_1=1,
        alpha_2=1,
    ):
        super(FusionBlock, self).__init__()

        self.alpha_central = alpha_central
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.fusion_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.fusion_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fusion_batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x1, x2, x_central=None):
        if x_central is None:
            x = self.alpha_1 * x1 + self.alpha_2 * x2
        else:
            x = self.alpha_central * x_central + self.alpha_1 * x1 + self.alpha_2 * x2

        x = self.fusion_conv(x)
        x = self.fusion_pool(x)
        x = self.relu(x)
        x = self.fusion_batch_norm(x)
        x = self.dropout(x)
        return x
