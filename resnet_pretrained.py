import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn


def get_resnet_pretrained():
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_classes = 29
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet
