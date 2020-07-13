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
# data_path = 'LTE_dataset_3c_1216.h5'
data_path = '/media/ubuntu/90679409-852b-4084-81e3-5de20cfa3035/Dwj/LTE_dataset_5c_1202.h5'
nrof_classes=3
seed = 666
batch_size = 20

val_classes = [0, 4, 3]
np.random.seed(seed=seed)
random.seed(seed)

with h5py.File(data_path, 'r') as h5f:
    feature_shape = h5f['features'].shape
    nrof_samples = feature_shape[0]


valid_indices = prepare.get_indices_by_classes(data_path, val_classes)



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

        nrof_classes = logits.shape[-1]
        print(nrof_classes)

        h5f = h5py.File(data_path, 'r')
        h5_features = h5f['features']
        h5_labels = h5f['labels']



        valid_labels = h5_labels[valid_indices]
        print([np.sum(valid_labels == i) for i in val_classes])


        # valid set

        valid_softmax = np.zeros((len(valid_indices), nrof_classes))

        epoch_size = int(np.ceil(len(valid_indices) / batch_size))

        for batch_number in range(epoch_size):

            batch_idx = slice(batch_number * batch_size, min(batch_number * batch_size + batch_size,len(valid_indices)))

            feed_dict = {phase_train_placeholder: False,
                         images_placeholder: h5_features[valid_indices[batch_idx]]}

            valid_softmax[batch_idx] = sess.run(softmax, feed_dict=feed_dict)

            if batch_number % 50 ==0:
                print(batch_number * batch_size)


        valid_predict = np.argmax(valid_softmax, 1).astype(np.int8)

        valid_labels = h5_labels[valid_indices]
        print([np.sum(valid_labels == i) for i in val_classes])
        for i in val_classes:
            idx = np.where(i == valid_labels)[0]
            # i = 0,3 or 4, real_label
            # idx = [1,2,3,4,5,6,7,...] where valid_labels[idx]=0 or valid_labels[idx]=2 or valid_labels[idx]=3
            # val_classes = [0,3,4]
            # valid_predict[idx] = 0,3 or 4
            print('class ', i,' acc:', np.mean(i == np.asarray(val_classes)[valid_predict[idx]]))

        h5f.close()
