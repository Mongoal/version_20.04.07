
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
from load import load_model
from data_loader.h5data_reader import H5DataReader
from utils.plot_feature import plot2d, plot, save_sig_fig
from utils.prepare_v02 import signal_regulation

# model_path = '../experiments/v1_sigmoid_cross_entropy/checkpoint'
model_path = '/media/ubuntu/ff94e166-11cc-4b03-9354-ac4e9f99c78e/lwj/version_20.04.07/experiments/v2_sigmoid_cross_entropy/checkpoint'
model_path = model_path
# stft_path = '../../dataset/LTE_dataset_stft_256x256x4_3c_1216.h5'
signal_path = '../../dataset/LTE_origin_3240_dataset_5c_10s_1202.h5'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#

with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load the model
        print(os.path.abspath(signal_path))

        ckpt = tf.train.get_checkpoint_state(model_path)
        load_model.load_model(model_path)
        # files = os.listdir(model_path)
        # meta_file = [s for s in files if s.endswith('.meta')][0]
        # saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file))

        # Get input and output tensors
        input = tf.get_default_graph().get_tensor_by_name("input:0")
        is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
        decode = tf.get_default_graph().get_tensor_by_name("decode:0")
        #回复模型
        # saver.restore(sess, ckpt.model_checkpoint_path)
        loss = tf.get_default_graph().get_tensor_by_name("loss/bce:0")

        # stft_reader = H5DataReader(stft_path,)
        sig_reader = H5DataReader(signal_path, 'r', 'signals',seed=303)
        # sig_reader = H5DataReader(stft_path, 'r',seed=30)
        batch, _ = sig_reader.get_shuffle_data(20)
        batch = np.asarray([signal_regulation(x) for x in batch])
        decode_arr ,los_array= sess.run([decode,loss],feed_dict={input:batch,is_training:False})
        diff = np.abs(decode_arr-batch)
        for i in range(len(decode_arr)):

            save_sig_fig(diff[i,:1000,0],  '../figures/sv2_diff_{:d}'.format(i))
            save_sig_fig(batch[i,:1000,0], '../figures/sv2_input_{:d}'.format(i))
            save_sig_fig(decode_arr[i,:1000,0],  '../figures/sv2_output_{:d}'.format(i))

        # np.exp(decode_arr) / sum(np.exp(decode_arr))
        # plot(np.decode_arr, 0)

