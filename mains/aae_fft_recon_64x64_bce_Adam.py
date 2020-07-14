import os
import sys
# TF_CPP_MIN_LOG_LEVEL = 1         //默认设置，为显示所有信息
# TF_CPP_MIN_LOG_LEVEL = 2         //只显示error和warining信息
# TF_CPP_MIN_LOG_LEVEL = 3         //只显示error信息
sys.path.append('..')
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from data_loader.data_generator import DataGenerator
from trainers.aae_trainer_2d import MyModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

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
    if config.loss=="hinge":
        from models.aae_hinge_bce_64x64_ADAM_sigmoid_model import AAEConv2dModel
    else:
        from models.aae_stft_bce_64x64_ADAM_sigmoid_model import AAEConv2dModel

    model = AAEConv2dModel(config)
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
