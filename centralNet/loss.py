import torch
import torch.nn as nn


def custom_loss(
    output_rgb,
    output_nir,
    output_central,
    target,
    alpha_mod1=1,
    alpha_mod2=1,
    alpha_central=1,
):
    criterion_cross_entropy = nn.CrossEntropyLoss()

    target = target.type(torch.LongTensor)
    loss1 = criterion_cross_entropy(output_rgb, target)
    loss2 = criterion_cross_entropy(output_nir, target)
    loss3 = criterion_cross_entropy(output_central, target)
    # return loss1 * 0.25 + loss2 * 0.25 + loss3 * 0.5
    return loss1 * alpha_mod1 + loss2 * alpha_mod2 + loss3 * alpha_central
