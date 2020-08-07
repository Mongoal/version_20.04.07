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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
            "CUDA_VISIBLE_DEVICES": "2",
            "exp_name": "resnet_softmax_128",
            "info": "5000 ->stft 128*128  model:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "input_shape": [128, 128, 4],
            "max_to_keep": 5
        }
        dict_v11={
            "nclass": 9,
            "model": "resnet",
            "CUDA_VISIBLE_DEVICES": "1",
            "exp_name": "resnet_softmax_128_v11",
            "info": "5000 ->stft 128*128  model:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "input_shape": [128, 128, 4],
            "max_to_keep": 5
        }
        dict_v11_a={
            "nclass": 9,
            "model": "resnet",
            "CUDA_VISIBLE_DEVICES": "1",
            "exp_name": "resnet_softmax_128_v11_a",
            "info": "resnet_softmax_128_v11改版a，更改学习率，信号去均值",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.003,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "input_shape": [128, 128, 4],
            "max_to_keep": 1
        }

        dict_v12={
            "nclass": 9,
            "model": "resnet_101",
            "CUDA_VISIBLE_DEVICES": "5",
            "exp_name": "resnet_softmax_128_v12",
            "info": "5000 ->stft 128*128  model:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "input_shape": [128, 128, 4],
            "max_to_keep": 5
        }

        dict_v13={
            "nclass": 9,
            "model": "resnet",
            "CUDA_VISIBLE_DEVICES": "7",
            "exp_name": "resnet_softmax_128_v13",
            "info": "5000 ->stft 128*128  output_stride=16 bn_decay 0.99 emodel:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "input_shape": [128, 128, 4],
            "max_to_keep": 5
        }
        dict_v14={
            "CUDA_VISIBLE_DEVICES": "7",
            "exp_name": "resnet_softmax_128_v14",
            "info": "5000 ->stft 128*128  output_stride=32 bn_decay 0.999 emodel:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "h5_shuffle_seed": 666,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "model": "resnet",
            "input_shape": [128, 128, 4],
            "nclass": 9,
            "output_stride": 32,
            "bn_decay": 0.999,
            "num_epochs": 200,
            "learning_rate": 0.001,
            "max_to_keep": 1
        }
        dict_v15={
            "CUDA_VISIBLE_DEVICES": "1",
            "exp_name": "resnet_softmax_128_v15",
            "info": "lr 0.0001 5000 ->stft 128*128  output_stride=32 bn_decay 0.997 emodel:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "h5_shuffle_seed": 666,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "model": "resnet",
            "input_shape": [128, 128, 4],
            "nclass": 9,
            "output_stride": 32,
            "bn_decay": 0.997,
            "num_epochs": 200,
            "learning_rate": 0.0001,
            "max_to_keep": 1
        }
        dict_v16={
            "CUDA_VISIBLE_DEVICES": "1",
            "exp_name": "resnet_softmax_128_v16",
            "info": "5000 ->stft 128*128  output_stride=32 bn_decay 0.997 emodel:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "h5_shuffle_seed": 666,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "model": "resnet_v1_50",
            "input_shape": [128, 128, 4],
            "nclass": 9,
            "output_stride": 32,
            "bn_decay": 0.997,
            "num_epochs": 200,
            "learning_rate": 0.0001,
            "max_to_keep": 1
        }
        dict_v17={
            "CUDA_VISIBLE_DEVICES": "3",
            "exp_name": "resnet_softmax_128_v17",
            "info": "5000 ->stft 128*128  output_stride=32 bn_decay 0.997 emodel:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "h5_shuffle_seed": 666,
            "batch_size": 64,
            "stft_method": "myfft1",
            "stft_args": (128, 128, 90, False),
            "model": "resnet_v1_50",
            "input_shape": [128, 128, 4],
            "nclass": 9,
            "output_stride": None,
            "bn_decay": 0.997,
            "num_epochs": 200,
            "learning_rate": 0.0001,
            "max_to_keep": 1
        }

        dict_v6={
            "CUDA_VISIBLE_DEVICES": "1",
            "exp_name": "resnet_softmax_128_v16",
            "info": "5000 ->stft 128*128  output_stride=32 bn_decay 0.997 emodel:resnet softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "h5_condition_args": (["fc"],[225],None),
            "h5_shuffle_seed": 666,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "model": "resnet_v1_50",
            "input_shape": [128, 128, 4],
            "nclass": 9,
            "output_stride": 32,
            "bn_decay": 0.997,
            "num_epochs": 200,
            "learning_rate": 0.001,
            "max_to_keep": 1
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
            "stft_args": (128, 128, 90, False),
            "input_shape": [128, 128, 4],
            "max_to_keep": 1
        }
        dict_v21={
            "nclass": 9,
            "model": "",
            "CUDA_VISIBLE_DEVICES": "5",
            "exp_name": "softmax_128_v21",
            "info": "5000 stft 128*128 model:softmax",
            "h5_data_path": "../dataset_signal_5000_new.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "stft_args": (128, 128, 90, False),
            "input_shape": [128, 128, 4],
            "max_to_keep": 1
        }
        dict_v3={
            "nclass": 9,
            "model": "resnet",
            "CUDA_VISIBLE_DEVICES": "2",
            "exp_name": "resnet_softmax_5000_256",
            "info": "5000 stft 256*256 model:softmax",
            "h5_data_path": "../dataset_signal_5000_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [256, 256, 4],
            "stft_args": (128,256,110,False), #(window,nfft,overlap,resize)
            "max_to_keep":1
        }

        dict_v4={
            "nclass": 9,
            "model": "resnet",
            "CUDA_VISIBLE_DEVICES": "2,3",
            "exp_name": "resnet_softmax_10000_256",
            "info": "10000 stft 256*256 model:softmax",
            "h5_data_path": "../dataset_signal_10000_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [256, 256, 4],
            "stft_args": (128,256,90,False), #(window,nfft,overlap,resize)
            "max_to_keep": 1
        }
        dict_v41={
            "nclass": 9,
            "model": "resnet_101",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "exp_name": "resnet_softmax_10000_256_v41",
            "info": "10000 stft 256*256 model:softmax",
            "h5_data_path": "../dataset_signal_10000_fc.h5",
            "h5_shuffle_seed": 666,
            "h5_data_key": "signals",
            "h5_label_key": "labels",
            "num_epochs": 200,
            "learning_rate": 0.001,
            "batch_size": 64,
            "input_shape": [256, 256, 4],
            "stft_args": (128,256,90,False), #(window,nfft,overlap,resize)
            "max_to_keep": 1
        }
        config = Bunch(dict_v6)
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
