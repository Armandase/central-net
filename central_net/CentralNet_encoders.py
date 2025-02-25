import torch
import torch.nn as nn
from decoders.MLP import MLP, MLP_2d
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
            nb_channels=nb_channels_mod2,
            nb_labels=num_classes,
            output_hidden_states=True,
        )

        fusion_layer1 = FusionBlock(
            in_channels=64,
            out_channels=256,
            stride=1,
            dropout_prob=dropout_rate,
        )
        fusion_layer2 = FusionBlock(
            in_channels=256,
            out_channels=512,
            stride=1,
            dropout_prob=dropout_rate,
            pooling=True,
        )
        fusion_layer3 = FusionBlock(
            in_channels=512,
            out_channels=1024,
            stride=2,
            dropout_prob=dropout_rate,
            pooling=False,
        )
        fusion_layer4 = FusionBlock(
            in_channels=1024,
            out_channels=2048,
            stride=2,
            dropout_prob=dropout_rate,
        )

        self.fusion_layers = [
            fusion_layer1,
            fusion_layer2,
            fusion_layer3,
            fusion_layer4,
        ]
        # self.classifier_mod1 = classifier
        # self.classifier_mod2 = classifier
        self.classifier_central = classifier

    def forward(self, x_mod1, x_mod2):
        classif_mod1, hiddens_out_mod1 = self.encoder_mod1(x_mod1)
        classif_mod2, hiddens_out_mod2 = self.encoder_mod2(x_mod2)

        central = None
        for mod1, mod2, fusion_layer in zip(
            hiddens_out_mod1, hiddens_out_mod2, self.fusion_layers
        ):
            central = fusion_layer(mod1, mod2, central)

        print("output central:", central.shape)
        # if self.classifier_mod1 is not None and self.classifier_mod2 is not None:
        # out_mod1 = self.classifier_mod1(out_mod1)
        # out_mod2 = self.classifier_mod2(out_mod2)
        classif_central = self.classifier_central(central)

        print("output central:", classif_central.shape)
        return classif_mod1, classif_mod2


def get_central_net(num_classes=6, channels_mod1=3, channels_mod2=1):
    # mlp = MLP(
    #     input_dim=2048,
    #     output_dim=num_classes,
    #     dropout_rate=0.3,
    # )
    mlp = MLP_2d(
        input_dim=2048,
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

    x_mod1 = torch.randn(3, nb_channel_mod1, 224, 224)
    x_mod2 = torch.randn(3, nb_channel_mod2, 224, 224)

    out_mod1, out_mod2 = central(x_mod1, x_mod2)
    print(out_mod1.shape, out_mod2.shape)
