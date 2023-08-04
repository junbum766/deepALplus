import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

class Loader(Dataset):
    def __init__(self, is_train=True, transform=None, path='/data/junbeom/data/cifar10/PT4AL'):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        if self.is_train: # train
            self.img_path = glob.glob('/data/junbeom/data/cifar10/PT4AL/train/*/*')
        else:
            self.img_path = glob.glob('/data/junbeom/data/cifar10/PT4AL/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])
        image_index = int(self.img_path[idx].split('/')[-1].split('.')[0])

        return img, label, image_index

class CIFAR10_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.classifier = nn.Linear(512, num_classes)

		self.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.dim = resnet18.fc.in_features

		
	
	def forward(self, x):

		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	def get_embedding_dim(self):
		return self.dim


# fix random seed
np.random.seed(433)
torch.manual_seed(433)
torch.backends.cudnn.enabled  = True
torch.backends.cudnn.benchmark= True

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")