import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt


class ConvBnReLu(nn.Module):
    '''Repetitive block of conv->batch norm->relu'''

    def __init__(self, in_planes, out_planes, drop=0.0, kernel=3, padding=1, stride=1):
        super(ConvBnReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                              stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.drop(self.relu(x))


def conv3x3(in_planes, out_planes, stride=1):
    '''Simple 3x3 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1)


class VGGNet(nn.Module):
    def __init__(self, num_feat=16, num_class=0):
        """Define the components of a VGG 11 model"""
        super(VGGNet, self).__init__()
        self.num_feat = num_feat
        self.conv_1 = conv3x3(3, num_feat)
        self.conv_2 = conv3x3(num_feat, 2 * num_feat)
        self.conv_3_1 = conv3x3(2 * num_feat, 4 * num_feat)
        self.conv_3_2 = conv3x3(4 * num_feat, 4 * num_feat)
        self.conv_4_1 = conv3x3(4 * num_feat, 8 * num_feat)
        self.conv_4_2 = conv3x3(8 * num_feat, 8 * num_feat)
        self.conv_5_1 = conv3x3(8 * num_feat, 8 * num_feat)
        self.conv_5_2 = conv3x3(8 * num_feat, 2 * num_feat)

        self.fc1 = nn.Linear((2 * num_feat) * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.pred = nn.Linear(256, num_class)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Input x is expected to be a 4d tensor (N x C X H x W)
           N - Number of images in minibatch
           C - Number of channels
           H,W  - Height and Width of image, respectively """

        conv_1 = self.conv_1(x)
        relu_1 = self.relu(conv_1)
        pool_1 = self.max_pool(relu_1)

        conv_2 = self.conv_2(pool_1)
        relu_2 = self.relu(conv_2)
        pool_2 = self.max_pool(relu_2)

        conv_3_1 = self.conv_3_1(pool_2)
        relu_3_1 = self.relu(conv_3_1)
        conv_3_2 = self.conv_3_2(relu_3_1)
        relu_3_2 = self.relu(conv_3_2)
        pool_3 = self.max_pool(relu_3_2)

        conv_4_1 = self.conv_4_1(pool_3)
        relu_4_1 = self.relu(conv_4_1)
        conv_4_2 = self.conv_4_2(relu_4_1)
        relu_4_2 = self.relu(conv_4_2)
        pool_4 = self.max_pool(relu_4_2)

        conv_5_1 = self.conv_5_1(pool_4)
        relu_5_1 = self.relu(conv_5_1)
        conv_5_2 = self.conv_5_2(relu_5_1)
        relu_5_2 = self.relu(conv_5_2)
        pool_5 = self.max_pool(relu_5_2)

        fc1 = self.relu(self.fc1(pool_5.view(-1, (2 * self.num_feat) * 7 * 7)))
        fc2 = self.relu(self.fc2(fc1))

        return self.pred(fc2)