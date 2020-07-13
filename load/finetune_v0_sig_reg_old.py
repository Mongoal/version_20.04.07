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
from load import load_model
from data_loader.h5data_reader import H5DataReader
from utils.logger import Logger
from utils.prepare_v02 import signal_regulation, signal_regulation_old

os.environ[ "CUDA_VISIBLE_DEVICES"]="1"
# model_path = '../experiments/v1_sigmoid_cross_entropy/checkpoint'
model_path = '../experiments/v0/checkpoint'
# model_path = '../experiments/v1_sigmoid_cross_entropy/checkpoint'
stft_path = '../../dataset/LTE_dataset_stft_256x256x4_3c_1216.h5'
signal_path = '../../dataset/LTE_origin_3240_dataset_5c_10s_1202.h5'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
exper_path = '../experiments/finetune/v0_sig_reg_old/'
finetune_model_dir = exper_path + 'model'
N = 3
batch_size = 50
init_lr = 0.001
iters = 100000
class Config:
    summary_dir = exper_path +'summary'

#
with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load the model
        print(os.path.abspath(signal_path))
        logger = Logger(sess, Config())
        ckpt = tf.train.get_checkpoint_state(model_path).model_checkpoint_path
        # files = os.listdir(model_path)
        meta_file = ckpt+'.meta'
        saver = tf.train.import_meta_graph(meta_file)

        # Get input and output tensors
        input = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
        is_training = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        labels = tf.placeholder(tf.int64,(None,),name = 'label')
        code = tf.get_default_graph().get_tensor_by_name("conv_7:0")
        decode = tf.get_default_graph().get_tensor_by_name("Tanh:0")
        with tf.variable_scope("Finetune"):
            flatten = tf.layers.flatten(code)
            net=tf.layers.dense(flatten, 128, activation='relu')
            logits = tf.layers.dense(net,N)
            softmax = tf.nn.softmax(logits)
        loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)
        predict = tf.arg_max(softmax,-1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,labels),tf.float32))
        train_varlist = tf.trainable_variables("Finetune")
        train_op = tf.train.AdamOptimizer(init_lr).minimize(loss,var_list=train_varlist)

        #began training
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)

        h5_reader = H5DataReader(signal_path,data_key='signals')

        i = 0
        los_list= []
        acc_list =[]
        while(iters > i ):
            i+= 1
            batch_x, batch_y = h5_reader.get_shuffle_data(batch_size)
            # preprocessing
            batch_x = [signal_regulation_old(x) for x in batch_x]
            feed_dict = {is_training: False, input: batch_x, labels: batch_y}
            _, prd, los, acc =sess.run([train_op,predict,loss,accuracy],feed_dict)
            los_list.append(los)
            acc_list.append(acc)


            if i%20 ==0:
                los = np.mean(los_list)
                acc = np.mean(acc_list)
                los_list = []
                acc_list = []
                print('step: [%d/%d]\tLoss %2.3f\tAcc %2.3f' %
                  (i , iters,  los, acc))
                summaries_dict = {
                    'loss_batch': los,
                    'acc': acc
                }
                logger.summarize(i+20000, summaries_dict=summaries_dict)

            if i%2000 == 0:
                if not os.path.exists(finetune_model_dir):
                    os.mkdir(finetune_model_dir)
                saver.save(sess, finetune_model_dir+'\ckpt',global_step=i)