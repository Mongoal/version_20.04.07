# 用来计算两张人脸图像之间的距离矩阵。需要输入的参数：
# 预训练模型 图片1  图片220170512-110547 1.png 2.png

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from datetime import datetime
import time
import random
import tensorflow as tf
import numpy as np
import h5py
import prepare
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
import inception_resnet_v1 as network
import re
import load_model


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
logs_base_dir = r'log'
models_base_dir = r'model'
data_path = 'LTE_dataset_3c_1216.h5'
# data_path = 'LTE_dataset_5c_1202.h5'
nrof_classes=3
seed = 666
batch_size = 20

np.random.seed(seed=seed)
random.seed(seed)

with h5py.File(data_path, 'r') as h5f:
    feature_shape = h5f['features'].shape
    nrof_samples = feature_shape[0]
    nrof_classes = int(h5f['labels'][:].max() + 1)
    print(nrof_classes)

train_indices, valid_indices = prepare.get_train_valid_indices(nrof_samples)
train_indices = np.sort(train_indices[:10000])
valid_indices = np.sort(valid_indices[:10000])
nrof_train_samples = train_indices.shape[0]
epoch_size = int(np.floor(nrof_train_samples / batch_size))



with tf.Graph().as_default():
    with tf.Session() as sess:
        # model = models_base_dir + '/20191203-000349'
        model = models_base_dir + '/20191217-120834'
        # Load the model
        load_model.load_model(model)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("images:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        logits = tf.get_default_graph().get_tensor_by_name("logits_1:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        softmax = tf.get_default_graph().get_tensor_by_name("Softmax:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


        h5f = h5py.File(data_path, 'r')
        h5_features = h5f['features']
        h5_labels = h5f['labels']



        valid_labels = h5_labels[valid_indices.tolist()]
        print([np.sum(valid_labels == i) for i in range(nrof_classes)])
        # train set
        train_softmax = np.zeros((len(train_indices), nrof_classes))

        epoch_size = int(np.ceil(len(train_indices) / batch_size))
        timer = time.time()
        for batch_number in range(epoch_size):
            batch_idx = range(batch_number * batch_size,
                              min(batch_number * batch_size + batch_size, len(train_indices)))

            feed_dict = {phase_train_placeholder: False,
                         images_placeholder: h5_features[train_indices[batch_idx].tolist()]}

            train_softmax[batch_idx] = sess.run(softmax, feed_dict=feed_dict)
            
            if batch_number % 50 ==0:
                print("step: %d ," % batch_number * batch_size, "time: %.1f s " % (time.time() - timer))
                timer = time.time()


        train_predict = np.argmax(train_softmax,1).astype(np.int8)

        train_labels = h5_labels[train_indices.tolist()]

        print('train acc:', np.mean(train_labels == train_predict))

        # valid set

        valid_softmax = np.zeros((len(valid_indices), nrof_classes))

        epoch_size = int(np.ceil(len(valid_indices) / batch_size))

        for batch_number in range(epoch_size):

            batch_idx = range(batch_number * batch_size, min(batch_number * batch_size + batch_size,len(valid_indices)))

            feed_dict = {phase_train_placeholder: False,
                         images_placeholder: h5_features[valid_indices[batch_idx].tolist()]}

            valid_softmax[batch_idx] = sess.run(softmax, feed_dict=feed_dict)

            if batch_number % 50 ==0:
                print(batch_number * batch_size)


        valid_predict = np.argmax(valid_softmax, 1).astype(np.int8)

        valid_labels = h5_labels[valid_indices.tolist()]
        # print([np.sum(valid_labels == i) for i in range(nrof_classes)])
        #
        # print('train acc:', np.mean(valid_labels == valid_predict))
        for i in [0, 1, 2]:
            idx = np.where(i == valid_labels)[0]
            print('class ', i, 'num:',len(idx),' acc:',  np.mean(valid_labels[idx] == valid_predict[idx]) )
        h5f.close()
