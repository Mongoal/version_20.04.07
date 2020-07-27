import tensorflow as tf
import os
import sys
sys.path.append('..')
from data_loader.data_generator import DataGenerator
from models.softmax_64x64_model import Conv2dModel
from trainers.softmax_trainer_2d import MyModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from bunch import Bunch


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        dict={
            "nclass": 9,
            "model": "resnet",
            "CUDA_VISIBLE_DEVICES": "1",
            "exp_name": "resnet_softmax_64x64",
            "info": "net 64x64 stft 128*128.resize model:softmax",
            "h5_data_path": "../dataset_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "features",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [64, 64, 4],
            "max_to_keep": 5
        }
        dict_v0={
            "nclass":9,
            "model": "",
            "CUDA_VISIBLE_DEVICES": "0",
            "exp_name": "softmax_64x64",
            "info": "3000 net 64x64 stft 128*128.resize model:softmax",
            "h5_data_path": "../dataset_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "features",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [64, 64, 4],
            "max_to_keep": 5
        }
        dict_v1={
            "nclass": 9,
            "model": "resnet",
            "CUDA_VISIBLE_DEVICES": "3",
            "exp_name": "resnet_softmax_128",
            "info": "5000 ->stft 128*128  model:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [128, 128, 4],
            "max_to_keep": 5
        }
        dict_v2={
            "nclass": 9,
            "model": "",
            "CUDA_VISIBLE_DEVICES": "3",
            "exp_name": "softmax_128",
            "info": "5000 stft 128*128 model:softmax",
            "h5_data_path": "../dataset_signal_5000_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [128, 128, 4],
            "max_to_keep": 5
        }
        dict_v3={
            "nclass": 9,
            "model": "",
            "CUDA_VISIBLE_DEVICES": "3",
            "exp_name": "softmax_128",
            "info": "5000 stft 256*256 model:softmax",
            "h5_data_path": "../dataset_signal_5000_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [128, 128, 4],
            "stft_args": (128,256,110,False), #(window,nfft,overlap,resize)
            "max_to_keep": 5
        }
        config = Bunch(dict_v2)
        config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/")
        config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
    except:
        print("missing or invalid arguments")
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you want
    model = Conv2dModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = MyModelTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
