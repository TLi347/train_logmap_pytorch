import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R
from pointnet_seg import classifier

def safe_norm(x, epsilon=1e-8, axis=None):
    return torch.sqrt(torch.maximum(torch.sum(x**2, axis=axis), torch.tensor(epsilon)))

# set network parameters here
BATCH_SIZE = 64
RESIZE=True
N_ORIG_NEIGHBORS = 200
N_NEIGHBORS_DATASET = 120
N_NEAREST_NEIGHBORS = 30
N_NEIGHBORS = 120
TESTING_SHAPES = [21, 11, 26]
TRAINING_SHAPES = list(set(list(range(56))) - set(TESTING_SHAPES))
N_TRAINING_SHAPES = len(TRAINING_SHAPES)
print(N_TRAINING_SHAPES)
N_TESTING_SHAPES = len(TESTING_SHAPES)
LOG_DIR = "log/log_famousthingi_classifier"
n_patches = 10000
path_records = "../data/training_data/famousthingi_logmap_patches_{}.tfrecords"

import torch
from tfrecord.torch.dataset import MultiTFRecordDataset

tfrecord_pattern = "../data/training_data/famousthingi_logmap_patches_{}.tfrecords"
index_pattern = "../data/training_index/{}.index"
splits = {
    "0": 0.017,
    "1": 0.017,
    "2": 0.017,
    "3": 0.017
}
description = {"patches": "float"}
dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

i=0
classifier_net = classifier(is_training=is_training, batch_size=BATCH_SIZE)
for (data) in iter(loader):
    data = data['patches'].reshape(-1,N_ORIG_NEIGHBORS,5)#(BATCH,120)
    rotations = torch.from_numpy(np.float32(R.random(BATCH_SIZE).as_matrix()))
    data = data[:,:N_NEIGHBORS]
    neighbor_points =data[:,:N_NEIGHBORS,:3]
    gt_map = data[:,:N_NEIGHBORS,3:]
    neighbor_points = neighbor_points - torch.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS, 1])
    neighbor_points = torch.transpose(torch.matmul(rotations, torch.transpose(neighbor_points, 2, 1)), 2, 1)#(none, 120, 3)
    
    if RESIZE:
        diag = safe_norm(torch.max(neighbor_points, axis = 1).values - torch.min(neighbor_points, axis = 1).values, axis = -1)
        diag = torch.tile(diag.view(2,1,1), [1, neighbor_points.shape[1], neighbor_points.shape[2]])
        neighbor_points = torch.div(neighbor_points, diag)#(BATCH, 120, 3)
        gt_map = torch.div(gt_map, diag[:,:,:2])#(BATCH, 120, 2)
    
    map = classifier_net(neighbor_points)#(BATCH, 120, 1)
    map = tf.squeeze(map)
    gt_dists = safe_norm(gt_map, axis = -1)

    geo_neighbors =tf.math.top_k(-gt_dists, k=N_NEAREST_NEIGHBORS)[1]
    geo_neighbors_indices =tf.stack([tf.tile(tf.range(BATCH_SIZE)[:, tf.newaxis], [ 1, N_NEAREST_NEIGHBORS]), geo_neighbors], axis = 2)
    geo_neighbors_indices = tf.reshape(geo_neighbors_indices, [-1, 2])
    labels = tf.cast(tf.scatter_nd(geo_neighbors_indices, tf.reshape(tf.ones_like(geo_neighbors), [-1]), map.shape), tf.float32)
    class_loss = tf.reduce_mean(tf.square(labels - tf.nn.sigmoid(map)))*30.0
    
    i=i+1
    if i>0:
        break


# TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
# VALSET_SIZE = n_patches*N_TESTING_SHAPES


# def data_load_pytorch():
#     with tf.device('/cpu:0'):
#             train_iterator, training_dataset =tf_dataset.dataset([path_records.format(k) for k in TRAINING_SHAPES],
#                                                                 batch_size=BATCH_SIZE, n_patches=n_patches, n_neighbors=N_ORIG_NEIGHBORS)

#             val_iterator, _ =tf_dataset.dataset([path_records.format(k) for k in TESTING_SHAPES]
#                             ,batch_size=BATCH_SIZE,n_patches=n_patches,n_neighbors=N_ORIG_NEIGHBORS)

#             handle = tf.placeholder(tf.string, shape=[])
#             iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
#             next = iterator.get_next() # THIS WILL BE USED AS OUR INPUT
#     with tf.device('/gpu:'+str(0)):
#         batch = tf.Variable(0)
#         learning_rate = tf.placeholder(tf.float32, shape=[])
#         is_training = tf.placeholder(tf.bool, shape=[])
#         rotations =tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3, 3])
#         data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,N_ORIG_NEIGHBORS*5])#(64,1000)
#         data = next[:,:N_ORIG_NEIGHBORS]#(none, 200, 5)

#         data = data[:,:N_NEIGHBORS]#(BATCH, 120, 5)
#         neighbor_points =data[:,:N_NEIGHBORS,:3]#(none, 120, 3)
#         gt_map = data[:,:N_NEIGHBORS,3:]#(none, 120, 2)
#         neighbor_points = neighbor_points - tf.tile(neighbor_points[:, 0:1, :], [1,N_NEIGHBORS, 1])#(none, 120, 3)
#         neighbor_points = tf.transpose(tf.matmul(rotations, tf.transpose(neighbor_points, [0, 2, 1])),[0, 2, 1])#(none, 120, 3)

#         if RESIZE:#normalization
#             diag =  safe_norm(tf.reduce_max(neighbor_points, axis = 1) - tf.reduce_min(neighbor_points, axis = 1), axis = -1)#(BATCH)
#             diag = tf.tile(diag[:,tf.newaxis,tf.newaxis], [1, neighbor_points.shape[1], neighbor_points.shape[2]])#(BATCH, 120, 3)
#             neighbor_points = tf.divide(neighbor_points, diag)#(BATCH, 120, 3)
#             gt_map = tf.divide(gt_map, diag[:,:,:2])#(BATCH, 120, 2)

