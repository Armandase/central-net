import torch
import torch.nn as nn
from decoders.MLP import MLP
from encoders.encoders import select_encoder
from utils import FusionBlock


class CentralNet(nn.Module):
    def __init__(
        self,
        num_classes,
        nb_channels_mod1,
        nb_channels_mod2,
        encoders="resnet50",
        classifier=None,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.encoder_mod1 = select_encoder(encoders)(
            nb_channels=nb_channels_mod1,
            nb_labels=num_classes,
            output_hidden_states=True,
        )
        self.encoder_mod2 = select_encoder(encoders)(
            nb_channels=nb_channels_mod1,
            nb_labels=num_classes,
            output_hidden_states=True,
        )
        self.fusion_layer1 = FusionBlock(
            in_channels=256,
            out_channels=512,
            stride=2,
            dropout_prob=dropout_rate,
        )
        self.fusion_layer2 = FusionBlock(
            in_channels=512,
            out_channels=256,
            stride=2,
            dropout_prob=dropout_rate,
        )
        self.fusion_layer3 = FusionBlock(
            in_channels=256,
            out_channels=512,
            stride=2,
            dropout_prob=dropout_rate,
        )
        self.fusion_layer4 = FusionBlock(
            in_channels=512,
            out_channels=512,
            stride=2,
            dropout_prob=dropout_rate,
        )

        self.classifier_mod1 = classifier
        self.classifier_mod2 = classifier

    def forward(self, x_mod1, x_mod2):
        classif_mod1, hiddens_out_mod1 = self.encoder_mod1(x_mod1)
        classif_mod2, hiddens_out_mod2 = self.encoder_mod2(x_mod2)

        # if self.classifier_mod1 is not None and self.classifier_mod2 is not None:
        # out_mod1 = self.classifier_mod1(out_mod1)
        # out_mod2 = self.classifier_mod2(out_mod2)

        return out_mod1, out_mod2


def get_central_net(num_classes=6, channels_mod1=3, channels_mod2=1):
    mlp = MLP(
        input_dim=512 * 4,
        output_dim=num_classes,
        dropout_rate=0.3,
    )
    model = CentralNet(
        nb_channels_mod1=channels_mod1,
        nb_channels_mod2=channels_mod2,
        num_classes=num_classes,
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
