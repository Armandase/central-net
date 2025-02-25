from .ResNet_timm import ResNetWithClassif, ResNet

encoders_list = {"resnet50_classif": ResNetWithClassif, "resnet50": ResNet}


def select_encoder(name):
    if name not in encoders_list:
        raise ValueError(
            f"Invalid fusion method name, options are: {list(encoders_list.keys())}"
        )
    return encoders_list[name]
