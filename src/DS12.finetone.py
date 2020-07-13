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
import load_model


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

logs_base_dir = r'log'
models_base_dir = r'model'
data_path = 'LTE_dataset_3c_1216.h5'


seed = 666
batch_size = 50
init_lr = 0.001
max_nrof_epochs = 100
transform = np.asarray([0,4,3])
np.random.seed(seed=seed)
random.seed(seed)

with h5py.File(data_path, 'r') as h5f:
    feature_shape = h5f['features'].shape
    nrof_samples = feature_shape[0]

    nrof_classes = int(h5f['labels'][:].max() + 1)
    print("NumClass",nrof_classes)

train_idx,valid_idx = prepare.get_train_valid_indices(nrof_samples,0.1,seed)
epoch_size = int(np.ceil(len(train_idx) / batch_size))
train_idx.sort()
valid_idx.sort()
print('make file train_idx.txt')
ftxt= open('train_idx.txt','w+')
print(*train_idx, sep='\n',file=ftxt)
ftxt.close()
print('make file valid_idx.txt')
ftxt= open('valid_idx.txt','w+')
print(*valid_idx, sep='\n',file=ftxt)
ftxt.close()
# train_indices, valid_indices = prepare.get_train_valid_indices(nrof_samples)
# train_indices = np.sort(train_indices)
# valid_indices = np.sort(valid_indices)
# nrof_train_samples = train_indices.shape[0]
# epoch_size = int(np.floor(nrof_train_samples / batch_size))
#
def train(sess,epoch,images,labels,phase_train,finetune_op,loss,acc,h5_features,h5_labels,softmax,file_log):
    batch_number = 0
    acclist =[]
    # Training loop
    shuffle_idx = np.random.permutation(len(train_idx))

    train_time = 0
    while batch_number < epoch_size:
        start_time = time.time()
        batch_idx = sorted(train_idx[shuffle_idx[batch_number * batch_size: batch_number * batch_size + batch_size]])
        feed_dict = {phase_train: False,
                     images: h5_features[batch_idx],
                     labels: transform[h5_labels[batch_idx]]}
        err, accuracy, _,  = sess.run([ loss, acc, finetune_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        acclist.append(accuracy)
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAcc %2.3f' %
              (epoch, batch_number + 1, epoch_size, duration, err, accuracy))
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAcc %2.3f' %
              (epoch, batch_number + 1, epoch_size, duration, err, accuracy),file=file_log)
        file_log.flush()
        batch_number += 1
        train_time += duration
    print('Epoch %d \tmean Acc %.3f \n#######################' %(epoch, np.mean(acclist)))


with tf.Graph().as_default():
    with tf.Session() as sess:
        model = r'model/20191203-000349'
        # Load the model
        ckpt = tf.train.get_checkpoint_state(model)
        load_model.load_model(model)
        files = os.listdir(model)
        meta_file = [s for s in files if s.endswith('.meta')][0]
        saver = tf.train.import_meta_graph(os.path.join(model, meta_file))

        # Get input and output tensors
        images = tf.get_default_graph().get_tensor_by_name("images:0")
        labels = tf.get_default_graph().get_tensor_by_name("labels:0")
        phase_train = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        logits = tf.get_default_graph().get_tensor_by_name("logits_1:0")
        softmax = tf.get_default_graph().get_tensor_by_name("Softmax:0")

        loss = tf.get_default_graph().get_tensor_by_name('total_loss:0')
        acc = tf.get_default_graph().get_tensor_by_name('Mean:0')
        global_step = tf.Variable(0, trainable=False)
        exp_lr = tf.train.exponential_decay(init_lr, global_step, batch_size, 0.9)
        finetune_variables = tf.trainable_variables("Logits")
        finetune_op = tf.train.AdamOptimizer(exp_lr, name="FTAdam").minimize(loss,global_step=global_step ,var_list=finetune_variables )
        init_var = []
        for var in tf.global_variables():
            if "FTAdam" in var.name:
                init_var.append(var)
        sess.run(tf.global_variables_initializer())

        #回复模型
        saver.restore(sess, ckpt.model_checkpoint_path)
        file_log = open('finetune_log.csv','w+')
        h5f = h5py.File(data_path, 'r')
        h5_features = h5f['features']
        h5_labels = h5f['labels']

        train(sess, -1, images, labels, phase_train, global_step, loss, acc, h5_features, h5_labels, softmax,
              file_log)
        epoch = 0
        while epoch < max_nrof_epochs:

            train(sess, epoch,images, labels, phase_train, finetune_op, loss, acc, h5_features, h5_labels,softmax,file_log)
            saver.save(sess, 'model/finetune', write_meta_graph=True)
            epoch += 1
        h5f.close()