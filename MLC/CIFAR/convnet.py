'''
ConvNet Models for CIFAR-10
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(128, num_classes)
        self.name = "ConvNet"

    def forward(self, x, return_h=False):
        out = self.features(x)
        hidden = out.view(out.size(0), -1)
        out = self.classifier(hidden)
        if return_h:
            return out, hidden
        else:
            return out

    def _make_layers(self):
        layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        return layers


def convnet(num_classes=10):
    model = ConvNet(num_classes)
    return model


class DeconvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DeconvNet, self).__init__()

        self.features = self._make_layers()
        self.classifier = nn.Linear(256, num_classes)
        self.name = "DeconvNet"

    def forward(self, x, return_h=False):
        out = self.features(x)
        hidden = out.view(out.size(0), -1)
        out = self.classifier(hidden)
        if return_h:
            return out, hidden
        else:
            return out

    def _make_layers(self):
        layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(96, 192, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.BatchNorm1d(4800),
            nn.Linear(4800, 256),
            nn.ReLU(inplace=True)
        )
        return layers

def deconvnet(num_classes=10):
    model = DeconvNet(num_classes)
    return model

