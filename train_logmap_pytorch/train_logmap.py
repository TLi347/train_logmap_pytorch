BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf_dataset
import pointnet_seg as logmap
import align

def safe_norm(x, epsilon=1e-8, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x) , axis=axis), epsilon))

def init_config():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    return config

# set network parameters here
BATCH_SIZE = 64
RESIZE=True
N_ORIG_NEIGHBORS = 200
N_NEIGHBORS_DATASET = 120
N_NEIGHBORS = 30
TESTING_SHAPES = [21, 11, 26]
TRAINING_SHAPES = list(set(list(range(56))) - set(TESTING_SHAPES))
N_TRAINING_SHAPES = len(TRAINING_SHAPES)
N_TESTING_SHAPES = len(TESTING_SHAPES)
LOG_DIR = "log/log_famousthingi_logmap"

n_patches = 10000
path_records = "../data/training_data/famousthingi_logmap_patches_{}.tfrecords"

TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
VALSET_SIZE = n_patches*N_TESTING_SHAPES

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from tfrecord.torch.dataset import MultiTFRecordDataset

tfrecord_pattern = "../data/training_data/famousthingi_logmap_patches_{}.tfrecords"
index_pattern = "../data/training_index/{}.index"
splits = {
    "0": 0.017,
    "1": 0.017,
    "2": 0.017,
    "3": 0.017,
}
description = {"patches": "float"}
train_dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

tfrecord_path = "../data/training_data/famousthingi_logmap_patches_0.tfrecords"
index_path = "../data/training_index/0.index"
test_dataset = TFRecordDataset(tfrecord_path, index_path, description)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

i=0
model = logmap(is_training=is_training, batch_size=BATCH_SIZE).cuda()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

def train(epoch):
    model.train()
    train_loss = 0
    for (data) in iter(train_loader):
        data = data['patches'].reshape(-1,N_ORIG_NEIGHBORS,5).cuda()#(BATCH,120)
        rotations = torch.from_numpy(np.float32(R.random(BATCH_SIZE).as_matrix()))
        data = data[:,:N_NEIGHBORS_DATASET]
        gt_map = data[:,:,3:]
        neighbor_points =data[:,:,:3]
        neighbor_points = neighbor_points - torch.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS_DATASET, 1])

        if RESIZE:
            diag =  safe_norm(torch.max(neighbor_points, axis = 1).values - torch.min(neighbor_points, axis = 1).values, axis = -1)
            diag += torch.empty(BATCH_SIZE).normal_(mean=0.0, std=0.01)
            diag = torch.tile(diag.view(BATCH_SIZE,1,1), [1, neighbor_points.shape[1], neighbor_points.shape[2]])
            neighbor_points = torch.div(neighbor_points, diag)#(BATCH, 120, 3)
            gt_map = torch.div(gt_map, diag[:,:,:2])#(BATCH, 120, 2)

        dists = safe_norm(gt_map, axis=-1)
        geo_neighbors = torch.topk(-dists, k=N_NEIGHBORS)[1]
        gt_map = torch.gather(gt_map, 1, geo_neighbors.unsqueeze(2).expand(-1,-1,2))
        neighbor_points = torch.gather(neighbor_points, 1, geo_neighbors.unsqueeze(2).expand(-1,-1,3))
        neighbor_points = neighbor_points - torch.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS, 1])
        neighbor_points = torch.matmul(rotations, neighbor_points.permute(0,2,1)).permute(0,2,1)

        map = model(neighbor_points)

        map = tf.convert_to_tensor(map.permute(0,2,1).detach())
        gt_map = tf.convert_to_tensor(gt_map.detach())
        map = align(map, gt_map)
        gt_map = tf.gather(gt_map, geo_neighbors, batch_dims=1)
        with tf.Session() as sess:
            gt_map = gt_map.eval()
            map = map.eval()
        gt_map = torch.from_numpy(gt_map)
        map = torch.from_numpy(map)

        dists = safe_norm(map, axis = -1)
        gt_dists = safe_norm(gt_map, axis = -1)
        loss_dist = torch.mean(torch.square(dists - gt_dists))
        loss_pos = torch.mean(torch.sum(torch.square(gt_map-map)))
        loss = loss_dist + loss_pos
        train_loss += loss.item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(trainloader.dataset)))




def test(epoch):
    model.eval()
    val_loss = []
    for (data) in iter(test_loader):
        data = data['patches'].reshape(-1,N_ORIG_NEIGHBORS,5).cuda()#(BATCH,120)
        rotations = torch.from_numpy(np.float32(R.random(BATCH_SIZE).as_matrix()))
        data = data[:,:N_NEIGHBORS_DATASET]
        gt_map = data[:,:,3:]
        neighbor_points =data[:,:,:3]
        neighbor_points = neighbor_points - torch.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS_DATASET, 1])

        if RESIZE:
            diag =  safe_norm(torch.max(neighbor_points, axis = 1).values - torch.min(neighbor_points, axis = 1).values, axis = -1)
            diag += torch.empty(BATCH_SIZE).normal_(mean=0.0, std=0.01)
            diag = torch.tile(diag.view(BATCH_SIZE,1,1), [1, neighbor_points.shape[1], neighbor_points.shape[2]])
            neighbor_points = torch.div(neighbor_points, diag)#(BATCH, 120, 3)
            gt_map = torch.div(gt_map, diag[:,:,:2])#(BATCH, 120, 2)

        dists = safe_norm(gt_map, axis=-1)
        geo_neighbors = torch.topk(-dists, k=N_NEIGHBORS)[1]
        gt_map = torch.gather(gt_map, 1, geo_neighbors.unsqueeze(2).expand(-1,-1,2))
        neighbor_points = torch.gather(neighbor_points, 1, geo_neighbors.unsqueeze(2).expand(-1,-1,3))
        neighbor_points = neighbor_points - torch.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS, 1])
        neighbor_points = torch.matmul(rotations, neighbor_points.permute(0,2,1)).permute(0,2,1)

        map = model(neighbor_points)

        map = tf.convert_to_tensor(map.permute(0,2,1).detach())
        gt_map = tf.convert_to_tensor(gt_map.detach())
        map = align(map, gt_map)
        gt_map = tf.gather(gt_map, geo_neighbors, batch_dims=1)
        with tf.Session() as sess:
            gt_map = gt_map.eval()
            map = map.eval()
        gt_map = torch.from_numpy(gt_map)
        map = torch.from_numpy(map)

        dists = safe_norm(map, axis = -1)
        gt_dists = safe_norm(gt_map, axis = -1)
        loss_dist = torch.mean(torch.square(dists - gt_dists))
        loss_pos = torch.mean(torch.sum(torch.square(gt_map-map)))
        loss = loss_dist + loss_pos
        val_loss.append(class_loss.item())
        
    print('\nValidation Completed!\tLoss: {:5.4f}'.format(np.asarray(val_loss).mean(0),))
    return np.asarray(val_loss).mean(0)




BEST_LOSS = 99999
LAST_SAVED = -1
N_EPOCHS = 10
for epoch in range(1, N_EPOCHS*10):
    print("Epoch {}:".format(epoch))
    train()
    cur_loss = test()
    if cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model!")
        torch.save(model.state_dict(), './checkpoint/classifier.pt')
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))