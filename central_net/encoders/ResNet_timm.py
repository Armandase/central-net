import torch
import torch.nn as nn
from transformers import ResNetConfig, ResNetModel, ResNetForImageClassification


# def get_resnet_timm(nb_channels, nb_labels=6, output_hidden_states=False):
#     print(ResNetConfig())
#     exit()
#     config = ResNetConfig(
#         num_channels=nb_channels,
#         num_labels=nb_labels,
#         output_hidden_states=output_hidden_states,
#     )
#     model = ResNetModel._from_config(config)
#     return model


# if __name__ == "__main__":
#     nb_channel = 3
#     nb_labels = 10

#     model = get_resnet_timm(nb_channel, nb_labels, True)
#     dummy_input = torch.randn(1, nb_channel, 224, 224)
#     output = model(dummy_input)
#     hidden = output.hidden_states
#     print("Hidden states:")
#     for a in hidden:
#         print(a.shape)
#     print("Classifier:")
#     print(output.last_hidden_state.shape)
#     # print(output)


from typing import Tuple, Optional, List


class ResNetWithClassif(nn.Module):
    def __init__(
        self, nb_channels: int, nb_labels: int, output_hidden_states: bool = False
    ):
        super().__init__()
        self.config = ResNetConfig(
            num_channels=nb_channels,
            num_labels=nb_labels,
            output_hidden_states=output_hidden_states,
        )
        self.resnet = ResNetModel._from_config(self.config)
        resnet_ouptput = 2048  # resnet output 2048
        self.classifier = nn.Linear(resnet_ouptput, nb_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        outputs = self.resnet(x)

        pooled = outputs.pooler_output.squeeze(-1).squeeze(-1)
        proba = self.softmax(self.classifier(pooled))
        return proba, (
            outputs.hidden_states if self.config.output_hidden_states else None
        )


class ResNet(nn.Module):
    def __init__(
        self, nb_channels: int, nb_labels: int, output_hidden_states: bool = False
    ):
        super().__init__()
        self.config = ResNetConfig(
            num_channels=nb_channels,
            num_labels=nb_labels,
            output_hidden_states=output_hidden_states,
        )
        self.resnet = ResNetModel._from_config(self.config)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        outputs = self.resnet(x)

        pooled = outputs.pooler_output.squeeze(-1).squeeze(-1)
        return pooled, (
            outputs.hidden_states if self.config.output_hidden_states else None
        )


if __name__ == "__main__":
    nb_channel = 3
    nb_labels = 10

    # model = ResNetWithClassif(nb_channel, nb_labels, True)
    model = ResNet(nb_channel, nb_labels, True)
    dummy_input = torch.randn(2, nb_channel, 224, 224)
    proba, hidden = model(dummy_input)

    print("Class probabilities:")
    print(proba.shape)

    print("Hidden states:")
    for layer_output in hidden:
        print(layer_output.shape)
