import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def addition(x1, x2):
    return torch.add(x1, x2)


def element_wise_multiplication(x1, x2):
    return torch.mul(x1, x2)


def concatenation(x1, x2):
    return torch.cat((x1, x2), dim=1)


# bilinear fusion
def dot_product(x1, x2):
    z = torch.matmul(x1, x2)
    return z


# bilinear fusion with a bias term
def augmented_dot_product(x1, x2):
    x1 = torch.cat(
        (x1, torch.ones(x1.shape[0], 1, x1.shape[-2], x1.shape[-1]).to(device)), dim=1
    )
    x2 = torch.cat(
        (x2, torch.ones(x2.shape[0], 1, x2.shape[-2], x2.shape[-1]).to(device)), dim=1
    )
    z = torch.matmul(x1, x2)
    return z


def polynomial_fusion(x1, x2):
    return x1 + x2 + x1 * x2


methods_list = {
    "add": addition,
    "mul": element_wise_multiplication,
    "concat": concatenation,
    "dot_product": dot_product,
    "augmented_dot_product": augmented_dot_product,
    "second_degree": polynomial_fusion,
}


def select_fusion_method(name):
    if name not in methods_list:
        raise ValueError(
            f"Invalid fusion method name, options are: {list(methods_list.keys())}"
        )
    return methods_list[name]


class FusionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        dropout_prob=0.4,
        fusion_method="add",
        pooling=False,
        alpha_central=1,
        alpha_1=1,
        alpha_2=1,
    ):
        super(FusionBlock, self).__init__()

        self.alpha_central = alpha_central
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.fusion_method = select_fusion_method(fusion_method)
        self.fusion_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        if pooling:
            self.fusion_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.fusion_pool = nn.Identity()
        self.relu = nn.ReLU()
        self.fusion_batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x1, x2, x_central=None):
        # if x_central is None:
        #     x = self.alpha_1 * x1 + self.alpha_2 * x2
        # else:
        #     x = self.alpha_central * x_central + self.alpha_1 * x1 + self.alpha_2 * x2
        if x_central is None:
            x = self.fusion_method(x1, x2)
        else:
            x = x_central + self.fusion_method(x1, x2)

        x = self.fusion_conv(x)
        x = self.fusion_pool(x)
        x = self.relu(x)
        x = self.fusion_batch_norm(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    x1 = torch.randn(1, 64, 224, 224).to(device)
    x2 = torch.randn(1, 64, 224, 224).to(device)

    add = select_fusion_method("add")
    b = add(x1, x2)
    print(b.shape)

    agm_dot = select_fusion_method("augmented_dot_product")
    c = agm_dot(x1, x2)
    print(c.shape)

    concat = select_fusion_method("concat")
    d = concat(x1, x2)
    print(d.shape)
