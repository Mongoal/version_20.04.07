# 用来计算两张人脸图像之间的距离矩阵。需要输入的参数：
# 预训练模型 图片1  图片220170512-110547 1.png 2.png

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
import os
from datetime import datetime
import time
import random
import tensorflow as tf
import numpy as np
from load import load_model
from data_loader.h5data_reader import H5DataReader
from utils.logger import Logger
from utils.prepare_v02 import myfft1_norm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

############################################
############### Exp Configs ################
############################################
os.environ[ "CUDA_VISIBLE_DEVICES"]="0"
## train set:
## condition key: ['labels', 'fc'], include: None, exclude: [(6, 225), (7, 225), (8, 225), (8, 300), (8, 380)]
exper_path = '../experiments/ft-ae_64x64_v5-ft_exp_2.0/'
model_path = '../experiments/fft_64x64_v5/checkpoint'
train_idx_txt = '../ft_exp_2.0.txt'
# model_path = '../experiments/v1_sigmoid_cross_entropy/checkpoint'
stft_path = '../dataset_fc.h5'
# signal_path = '../../dataset/LTE_origin_3240_dataset_5c_10s_1202.h5'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
finetune_model_dir = exper_path + 'model'
N = 8
batch_size = 32
init_lr = 0.001
iters = 200000
class Config:
    summary_dir = exper_path +'summary'

#
with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load the model
        print(os.path.abspath(stft_path))
        logger = Logger(sess, Config())
        ckpt = tf.train.get_checkpoint_state(model_path).model_checkpoint_path
        # files = os.listdir(model_path)
        meta_file = ckpt+'.meta'
        tf.train.import_meta_graph(meta_file)
        saver = tf.train.Saver(max_to_keep=1)

        # Get input and output tensors

        # Get input and output tensors
        input = tf.get_default_graph().get_tensor_by_name("input:0")
        is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
        decode = tf.get_default_graph().get_tensor_by_name("decode:0")
        loss = tf.get_default_graph().get_tensor_by_name("loss/cross_entropy:0")
        labels = tf.placeholder(tf.int64,(None,),name = 'label')
        code = tf.get_default_graph().get_tensor_by_name("code:0")
        print("code.shape : ",code.shape)
        finetune_step = tf.Variable(0,trainable=False)

        with tf.variable_scope("Finetune"):
            flatten = tf.layers.flatten(code)
            net=tf.layers.dense(flatten, 1024, activation='relu')
            logits = tf.layers.dense(net,N)
            softmax = tf.nn.softmax(logits)

        loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)
        predict = tf.arg_max(softmax,-1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,labels),tf.float32))
        train_varlist = tf.trainable_variables("Finetune")
        train_op = tf.train.AdamOptimizer(init_lr).minimize(loss,var_list=train_varlist,global_step=finetune_step)
        #began training
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)

        h5_reader = H5DataReader(stft_path,seg_set_method='txt',txt_path=train_idx_txt)
        unknown_reader = H5DataReader(stft_path)
        unknown_reader.set_condition_idx(condition_keys=['labels', 'fc'], include=[(6, 225), (7, 225), (8, 225)],exclude_conditions=None)
        i = 0
        los_list= []
        acc_list =[]
        e_los_list= []
        e_acc_list =[]
        u_los_list= []
        u_acc_list =[]
        while(iters > i ):
            i+= 1
            batch_x, batch_y = h5_reader.get_train_batch(batch_size)
            feed_dict = {is_training: False, input: batch_x, labels: batch_y}
            _, prd, los, acc =sess.run([train_op,predict,loss,accuracy],feed_dict)
            los_list.append(los)
            acc_list.append(acc)

            batch_x, batch_y = h5_reader.get_test_batch(batch_size)
            feed_dict = {is_training: False, input: batch_x, labels: batch_y}
            e_prd, e_los, e_acc =sess.run([predict,loss,accuracy],feed_dict)
            e_los_list.append(los)
            e_acc_list.append(acc)

            batch_x, batch_y = unknown_reader.get_shuffle_data(batch_size)
            feed_dict = {is_training: False, input: batch_x, labels: batch_y}
            u_prd, u_los, u_acc =sess.run([predict,loss,accuracy],feed_dict)
            u_los_list.append(los)
            u_acc_list.append(acc)



            if i%20 ==0:
                los = np.mean(los_list)
                acc = np.mean(acc_list)
                e_los = np.mean(e_los_list)
                e_acc = np.mean(e_acc_list)
                u_los = np.mean(u_los_list)
                u_acc = np.mean(u_acc_list)
                los_list.clear()
                acc_list.clear()
                e_los_list.clear()
                e_acc_list.clear()
                u_los_list.clear()
                u_acc_list.clear()
                print('step: [%d/%d]\tLoss %2.3f\tAcc %2.3f\tEval Loss %2.3f\tEval Acc %2.3f\tunknown Loss %2.3f\tunknown Acc %2.3f' %
                  (i , iters,  los, acc, e_los, e_acc, u_los, u_acc))

                summaries_dict = {
                    'los': los,
                    'acc': acc,
                    'e_los':e_los,
                    'e_acc':e_acc,
                    'u_los':u_los,
                    'u_acc':u_acc,
                }
                logger.summarize(sess.run(finetune_step), summaries_dict=summaries_dict)

            if i%2000 == 0:
                if not os.path.exists(finetune_model_dir):
                    os.mkdir(finetune_model_dir)
                saver.save(sess, finetune_model_dir,global_step=i)
