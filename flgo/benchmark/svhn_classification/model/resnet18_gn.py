"""
This model is copied from the github repo of FedDyn
(https://github.com/alpemreacar/FedDyn/blob/master/utils_models.py)
"""

from flgo.utils import fmodule
import torchvision.models
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

class Model(fmodule.FModule):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18()
        resnet18.fc = nn.Linear(512, 10)
        resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.model = resnet18

    def forward(self, x):
        return self.model(x)

class AugmentDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = torchvision.transforms.Compose([RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(0.5)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        return self.transform(img), label

def init_dataset(object):
    if 'Client' in object.get_classname():
        object.train_data = AugmentDataset(object.train_data)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)