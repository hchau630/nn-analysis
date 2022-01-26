import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

__all__ = ['resnet50_1c']

def resnet50_1c(**kwargs):
    model = models.resnet50(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.preprocess = transforms.Grayscale()
    model.register_forward_pre_hook(lambda _, inp: model.preprocess(inp[0]))
    return model