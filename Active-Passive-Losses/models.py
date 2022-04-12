import torch
import torch.nn as nn
import torch.nn.functional as F
import mlconfig
import torchvision
mlconfig.register(torchvision.models.resnet50)
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        return self.out_conv(x)


@mlconfig.register
class ToyModel(nn.Module):
    def __init__(self, type='CIFAR10'):
        super(ToyModel, self).__init__()
        self.type = type
        if type == 'CIFAR10':
            self.block1 = nn.Sequential(
                ConvBrunch(3, 64, 3),
                ConvBrunch(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.block2 = nn.Sequential(
                ConvBrunch(64, 128, 3),
                ConvBrunch(128, 128, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.block3 = nn.Sequential(
                ConvBrunch(128, 196, 3),
                ConvBrunch(196, 196, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(
                nn.Linear(4*4*196, 256),
                nn.BatchNorm1d(256),
                nn.ReLU())
            self.fc2 = nn.Linear(256, 10)
            self.fc_size = 4*4*196
        elif type == 'MNIST':
            self.block1 = nn.Sequential(
                ConvBrunch(1, 32, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.block2 = nn.Sequential(
                ConvBrunch(32, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2))
            # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(
                nn.Linear(64*7*7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU())
            self.fc2 = nn.Linear(128, 10)
            self.fc_size = 64*7*7
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) if self.type == 'CIFAR10' else x
        # x = self.global_avg_pool(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, type='CIFAR10'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if type == 'Pneumonia':
            self.linear = nn.Linear(512*block.expansion*16, num_classes)
        else:
            self.linear = nn.Linear(512*block.expansion, num_classes)
        self._reset_prams()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return


#@mlconfig.register
#def ResNet18(num_classes=10):
#    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


#@mlconfig.register
#def ResNet34(num_classes=10):
#    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


@mlconfig.register
class ResNet18(ResNet):
    def __init__(self, type='CIFAR10', num_classes=10):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes, type)


@mlconfig.register
class ResNet50(ResNet):
    def __init__(self, type='CIFAR10', num_classes=10):
        super(ResNet50, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes, type)


#@mlconfig.register
#def ResNet50(num_classes=10):
#    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


#@mlconfig.register
#def ResNet101(num_classes=10):
#    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


#@mlconfig.register
#def ResNet152(num_classes=10):
#    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, type='CIFAR10', num_classes=10):
        super(VGG, self).__init__()

        if type=='Pneumonia':
            in_channels = 3
            self.features = self._make_layers(cfg[vgg_name], in_channels)
            self.classifier = nn.Linear(512*16, num_classes)
        else:
            in_channels = 3
            self.features = self._make_layers(cfg[vgg_name], in_channels)
            self.classifier = nn.Linear(512, num_classes)
        self.name = vgg_name

    def forward(self, x, return_h=False):
        out = self.features(x)
        hidden = out.view(out.size(0), -1)
        out = self.classifier(hidden)
        if return_h:
            return out, hidden
        else:
            return out

    def _make_layers(self, cfg, in_channels=3):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


@mlconfig.register
class VGG11(VGG):
    def __init__(self, type='CIFAR10', num_classes=10):
        super(VGG11, self).__init__('VGG11', type, num_classes)


@mlconfig.register
class VGG16(VGG):
    def __init__(self, type='CIFAR10', num_classes=10):
        super(VGG16, self).__init__('VGG16', type, num_classes)


@mlconfig.register
class ConvNet(nn.Module):
    def __init__(self, type='CIFAR10', num_classes=10):
        super(ConvNet, self).__init__()

        if type=='Pneumonia':
            self.features = self._make_pneu_layers()
        else:
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

    def _make_pneu_layers(self):
        layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
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
            nn.BatchNorm1d(25088),
            nn.Linear(25088, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        return layers


@mlconfig.register
class DeconvNet(nn.Module):
    def __init__(self, type='CIFAR10', num_classes=10):
        super(DeconvNet, self).__init__()

        if type=='Pneumonia':
            self.features = self._make_pneu_layers()
        else:
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

    def _make_pneu_layers(self):
        layers = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(3, 3)),
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
            nn.BatchNorm1d(161472),
            nn.Linear(161472, 256),
            nn.ReLU(inplace=True)
        )
        return layers


class MobileBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


@mlconfig.register
class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, type='CIFAR10', num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        if type == 'Pneumonia':
            self.linear = nn.Linear(1024*16, num_classes)
        else:
            self.linear = nn.Linear(1024, num_classes)
        self.name = "MobileNet"

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(BasicBlock(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, return_h=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        if return_h:
            return out, hidden
        else:
            return out
