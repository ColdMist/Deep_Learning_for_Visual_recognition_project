import torch.utils.data as data
from data_utilities import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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
from models.vgg import *
from collections import defaultdict
#from IPython.display import clear_output
from sklearn.preprocessing import LabelEncoder


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
BASE_PATH = 'dataset/food-101'
#Some hyper params
TEST_SPLIT = 0.25
epochs = 80
batch_size = 16
SAMPLE_TRAINING = False # make train set smaller for faster iteration
IMG_SIZE = (224, 244)
LEARNING_RATE = 3e-4

# Get all the version of training sets as mentioned in the project
all_img_df_10_1, all_img_df_10_2, all_img_df_10_3, all_img_df_20_1, all_img_df_20_2 = generate_data_sets(BASE_PATH)

#train the intended version of dataset you want to train (e.g. sampled 10 labeled sets and sample 20 labeled set)
all_img_df = all_img_df_10_1 # This is the first sample of 10 labels out of 101 labels


cat_enc = LabelEncoder()
all_img_df ['cat_idx'] = cat_enc.fit_transform(all_img_df ['category'])
N_CLASSES = len(cat_enc.classes_)
# replace with random labels
all_img_df ['cat_idx'] = np.random.choice(range(N_CLASSES), size=all_img_df .shape[0])
print(N_CLASSES, 'classes')
#all_img_df.sample(5)

train_df, test_df = train_test_split(all_img_df,
                                     test_size=TEST_SPLIT,
                                     random_state=42,
                                     stratify=all_img_df['category'])

# if SAMPLE_TRAINING: # make train smaller for faster testing
#     train_df = train_df.\
#         groupby('category').\
#         apply(lambda x: x.sample(50)).\
#         reset_index(drop=True).\
#         sample(frac=1).\
#         reset_index(drop=True)
# print('train', train_df.shape[0], 'test', test_df.shape[0])


train_dataset = DataWrapper(train_df, True, IMG_SIZE)
train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,
            batch_size=batch_size, pin_memory=False)#, num_workers=4)

test_dataset = DataWrapper(test_df, True, IMG_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True,
            batch_size=batch_size, pin_memory=False) #num_workers=1)


model = VGGNet(num_class=N_CLASSES).to(device) #VGG style model
criterion = nn.CrossEntropyLoss() #Use cross entropy loss

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

use_gpu = True



train_results = defaultdict(list)
train_iter, test_iter, best_acc = 0, 0, 0
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.set_title('Train Loss')
ax2.set_title('Train Accuracy')
ax3.set_title('Test Loss')
ax4.set_title('Test Accuracy')

for i in range(epochs):
    #clear_output(wait=True)
    plt.show()
    print("Epoch ", i)
    ## Train Phase
    # Model switches to train phase
    model.train()

    # Running through all mini batches in the dataset
    count, loss_val, correct, total = train_iter, 0, 0, 0
    for data, target in tqdm(train_loader, desc='Training'):
        if use_gpu:  # Using GPU & Cuda
            data, target = data.to(device), target.to(device)

        output = model(data)  # FWD prop
        loss = criterion(output, target)  # Cross entropy loss
        c_loss = loss.data.item()
        ax1.plot(count, c_loss, 'r.')
        loss_val += c_loss

        optimizer.zero_grad()  # Zero out any cached gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

        # Compute accuracy
        predicted = output.data.max(1)[1]  # get index of max
        total += target.size(0)  # total samples in mini batch
        c_acc = (predicted == target).sum().item()
        ax2.plot(count, c_acc / target.size(0), 'r.')
        correct += c_acc
        count += 1
    train_loss_val, train_iter, train_acc = loss_val / len(train_loader.dataset), count, correct / float(total)

    print("Training loss: ", train_loss_val, " train acc: ", train_acc)
    ## Test Phase

    # Model switches to test phase
    model.eval()

    # Running through all mini batches in the dataset
    count, correct, total, lost_val = test_iter, 0, 0, 0
    for data, target in tqdm(test_loader, desc='Testing'):
        if use_gpu:  # Using GPU & Cuda
            data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)  # Cross entropy loss
        c_loss = loss.data.item()
        ax3.plot(count, c_loss, 'b.')
        loss_val += c_loss
        # Compute accuracy
        predicted = output.data.max(1)[1]  # get index of max
        total += target.size(0)  # total samples in mini batch
        c_acc = (predicted == target).sum().item()
        ax4.plot(count, c_acc / target.size(0), 'b.')
        correct += c_acc
        count += 1

    # Accuracy over entire dataset
    test_acc, test_iter, test_loss_val = correct / float(total), count, loss_val / len(test_loader.dataset)
    print("Epoch: ", i, " test set accuracy: ", test_acc)

    train_results['epoch'].append(i)
    train_results['train_loss'].append(train_loss_val)
    train_results['train_acc'].append(train_acc)
    train_results['train_iter'].append(train_iter)

    train_results['test_loss'].append(test_loss_val)
    train_results['test_acc'].append(test_acc)
    train_results['test_iter'].append(test_iter)

    # Save model with best accuracy
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

fig.savefig('train_curves.png')


