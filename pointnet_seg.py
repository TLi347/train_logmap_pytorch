import torch
import torch.nn as nn
from collections import OrderedDict

import numpy as np
import math
import sys
import os

def safe_norm(x, epsilon=1e-8, axis=None):
    return torch.sqrt(torch.maximum(torch.sum(x**2, axis=axis), torch.tensor(epsilon)))

class classifier(nn.Module):
    def __init__(self, is_training,  batch_size=8):
        super(classifier, self).__init__()
        
        self.is_training = is_training
        self.batch_size = batch_size

        self.layer1 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(1,64,kernel_size=(1,4), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(64,64,kernel_size=(1,1), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(64,128,kernel_size=(1,1), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer4 = nn.Sequential(OrderedDict([
                    ('c', torch.nn.Conv2d(128,256,kernel_size=(1,1), padding=0, stride=1)),
                    ('relu', nn.ReLU())
        ]))
        self.layer5 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(256,1024,kernel_size=(1,1), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer6 = nn.MaxPool2d((120,1),stride=(2,2))
        self.layer7 = nn.Linear(1024,1024)
        self.layer8 = nn.Linear(1024,1024)
        self.layer9 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(1028,1024,kernel_size=(1,1), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer10 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(1024,528,kernel_size=(1,1), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer11 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(528,3,kernel_size=(1,1), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer12 = nn.Sequential(OrderedDict([
            ('c', torch.nn.Conv2d(1028,1024,kernel_size=(1,1), padding=0, stride=1)),
            ('relu', nn.ReLU())
        ]))
        self.layer13 = nn.Sequential(OrderedDict([
                    ('c', torch.nn.Conv2d(1024,528,kernel_size=(1,1), padding=0, stride=1)),
                    ('relu', nn.ReLU())
                ]))
        self.layer14 = nn.Sequential(OrderedDict([
                    ('c', torch.nn.Conv2d(528,1,kernel_size=(1,1), padding=0, stride=1)),
                    ('relu', nn.ReLU())
                ]))
    
    def forward(self, point_cloud):
        # point_cloud = point_cloud.cuda()
        num_point = point_cloud.shape[1]
        euc_dists = safe_norm(point_cloud - torch.tile(point_cloud[:,0:1,:], [1,point_cloud.shape[1], 1]), axis = -1).unsqueeze(2)
        point_cloud = torch.cat([point_cloud,euc_dists],axis=2)
        input_image = torch.unsqueeze(point_cloud, 1)

        x = self.layer1(input_image)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        points_feat1 = self.layer5(x)
        pc_feat1 = self.layer6(points_feat1)
        pc_feat1 = pc_feat1.reshape(self.batch_size, 1024)
        pc_feat1 = self.layer7(pc_feat1)
        pc_feat1 = self.layer8(pc_feat1)
        pc_feat1_expand = pc_feat1.reshape(8, -1, 1, 1)
        pc_feat1_expand = torch.tile(pc_feat1_expand, [1, 1, num_point, 1])
        points_feat1_concat = torch.cat((point_cloud.unsqueeze(2).permute(0,3,1,2),pc_feat1_expand),axis=1)
        x = self.layer9(points_feat1_concat)
        x = self.layer10(x)
        x = self.layer11(x)

        euc_dists = safe_norm(x - torch.tile(x[:,:,0:1,:], [1, 1, point_cloud.shape[1], 1]), axis = 1).unsqueeze(1)
        pc1 = x
        x = torch.cat((x, euc_dists),axis=1)

        points_feat2_concat = torch.cat((x, pc_feat1_expand),axis=1)
        x = self.layer12(points_feat2_concat)
        x = self.layer13(x)
        x = self.layer14(x)
        x = x.squeeze(1)

        return x


array = np.arange(2880).reshape(8,120,3)
array = np.float32(array)
point_cloud = torch.from_numpy(array)
net = classifier(is_training=True, batch_size=8)
output = net(point_cloud)
print(output.shape)
