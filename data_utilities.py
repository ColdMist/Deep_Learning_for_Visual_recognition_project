from models.vgg import *
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

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
BASE_PATH = 'dataset/food-101'
MICRO_DATA = True
IMG_SIZE = (224, 244)

class DataWrapper(data.Dataset):
    ''' Data wrapper for pytorch's data loader function '''
    def __init__(self, image_df, resize, IMG_SIZE=(224,244)):
        self.dataset = image_df
        self.resize = resize
        self.IMG_SIZE = IMG_SIZE

    def __getitem__(self, index):
        c_row = self.dataset.iloc[index]
        image_path, target = c_row['path'], c_row['cat_idx']  #image and target
        #read as rgb image, resize and convert to range 0 to 1
        image = cv2.imread(image_path, 1)
        if self.resize:
            image = cv2.resize(image, self.IMG_SIZE)/255.0
        else:
            image = image/255.0
        image = (torch.from_numpy(image.transpose(2,0,1))).float() #NxCxHxW
        return image, int(target)

    def __len__(self):
        return self.dataset.shape[0]

def generate_data_sets(BASE_PATH):
    all_img_df = pd.DataFrame({'path':
                            glob(os.path.join(BASE_PATH, 'images', '*', '*.jpg'))})
    all_img_df['category'] = all_img_df['path'].map(lambda x:
                            os.path.split(os.path.dirname(x))[-1].replace('_', ' '))

    all_categories = list(set(all_img_df['category']))

    category_set_10_1 = np.random.choice(all_categories, 10)
    category_set_10_2 = np.random.choice(all_categories, 10)
    category_set_10_3 = np.random.choice(all_categories, 10)

    category_set_20_1 = np.random.choice(all_categories, 20)
    category_set_20_2 = np.random.choice(all_categories, 20)


    if MICRO_DATA:
        all_img_df_10_1 = all_img_df[all_img_df['category'].isin(category_set_10_1)]
        all_img_df_10_2 = all_img_df[all_img_df['category'].isin(category_set_10_2)]
        all_img_df_10_3 = all_img_df[all_img_df['category'].isin(category_set_10_3)]

        all_img_df_20_1 = all_img_df[all_img_df['category'].isin(category_set_20_1)]
        all_img_df_20_2 = all_img_df[all_img_df['category'].isin(category_set_20_2)]

    return all_img_df_10_1, all_img_df_10_2, all_img_df_10_3, all_img_df_20_1, all_img_df_20_2