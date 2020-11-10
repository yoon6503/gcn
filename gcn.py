import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
from PIL import Image
import random
import math

from __future__ import division
from __future__ import print_function

import argparse

import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

numalph = 0
#unzip ~/PycharmProjects/test

#train data
chardict = {} #[alphabet: di ct, ...]
# charname = [] #list of alphabet

root = os.path.abspath("/home/yoon1524/PycharmProjects/test/images_background")
charname = os.listdir(root) #English, Korean .... string type
for alphname in charname: #create train set
    charroot = os.path.join(root, alphname)
    charroot = os.path.abspath(charroot)
    chartype = os.listdir(charroot) #character01, ...
    typedict = {}
    # typedictint = {} #int version of typedict
    numtype = 0
    for dirname in chartype:
        typedict[dirname] = os.listdir(os.path.join(charroot, dirname))
        # typedictint[numtype] = os.listdir(os.path.join(charroot, dirname))
        numtype += 1
    chardict[alphname] = typedict
    numalph += 1

class Omniglot(Dataset):

    #datafile: ex) English, a
    def __init__(self, data_file, sp_data_file, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_file = data_file
        self.sp_data_file = sp_data_file

    def __len__(self):
        return len(self.sp_data_file)

    def __getitem__(self, idx):
        # idx : num of char type (range(20))
        image_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])
        dirlist = self.root_dir.split(os.path.sep)
        alphabet = dirlist[-2]
        character = dirlist[-1]
        img = Image.open(image_path).resize((28, 28))
        img = img.convert('L')
        x = torch.from_numpy(np.asarray(img))
        y = torch.zeros((1,2))
        for i, (lang, chard) in enumerate(self.data_file.items()):
           if lang == alphabet:
                y[0,0] = i
                for j, char in enumerate(chard):
                    if char == character:
                        y[0,1] = j
                        break
        if self.transform:
            x = self.transform(x)
        return x, y

def datasampler(omnig, num_sample):
    idxlist = np.random.choice(omnig.__len__(), num_sample, replace=False)
    x_sample = torch.zeros((num_sample,)+omnig.__getitem__(0)[0].shape)
    y_sample = torch.zeros((num_sample, 1, 2))
    i = 0
    for idx in idxlist:
        x_sample[i:], y_sample[i:] = omnig.__getitem__(idx)
        i += 1
    x_sample = torch.reshape(x_sample, (28*28,1))
    return x_sample, y_sample

def randomclass(dataset, num_class):
    random_class = []
    for i in range(num_class):
        alphabet = random.choice(list(dataset.items()))
        _character = random.choice(list(alphabet[1].items()))
        class_tuple = (alphabet[0], _character[0])
        while class_tuple in random_class:
            alphabet = random.choice(list(dataset.items()))
            _character = random.choice(list(alphabet[1].items()))
            class_tuple = (alphabet[0], _character[0])
        random_class.append(class_tuple)
    return random_class

random_class = randomclass(chardict, 1)
root_dir = os.path.join(root, random_class[0][0], random_class[0][1])
sp_data_file = os.listdir(root_dir)

dataset = Omniglot(chardict, sp_data_file, root_dir)

class GraphConvolution(Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

features, labels, idx_train = data_sampler(dataset, num), torch.arange(num)  #we need to set num 
adj = torch.zeros(28*28,28*28)   #we need to set adjacency matrix(user's choice)

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
# Train model
for i in range(epoch):
    train()
print("Optimization Finished!")