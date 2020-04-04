import torch
import torch.nn as nn
from torchvision import models


def vgg16(out_dims, fine_tuning=False , device=torch.device('cpu')):
    """
    Model vgg de pytorch pret a etre finetuned/ transfer learnt
    """

    vgg16 = models.vgg16(pretrained=True).to(device)

    # Finetuning
    for param in vgg16.features.parameters():
        param.require_grad = fine_tuning
        param = param.to(device)

    for param in vgg16.classifier.parameters():
        param.require_grad = True
        param = param.to(device)
    # replacing the last layer
    vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True).to(device)

    return vgg16
