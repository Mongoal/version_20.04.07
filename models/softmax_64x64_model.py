from base.base_model import BaseModel
import tensorflow as tf
from models.cnn_utils import *
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from models import resnet_v1
slim = tf.contrib.slim



class Conv2dModel(BaseModel):
    def __init__(self, config):
        super(Conv2dModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # 默认参数
        FILTER_SIZE = (5, 5)
        Z_DIM = 2048
        STRIDE = (2, 2)
        DEPTHS = [64, 128, 256, 256, 128, 32]
        CHANNELS = 4
        N_CLASS = self.config.nclass
        def encoder(input, z_dim=Z_DIM, is_training=False):
            net = conv2d_BN(input, DEPTHS[0], FILTER_SIZE, is_training, stride=STRIDE, name='conv_1',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN(net, DEPTHS[1], FILTER_SIZE, is_training, stride=STRIDE, name='conv_2',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN(net, DEPTHS[2], FILTER_SIZE, is_training, stride=STRIDE, name='conv_3',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            z = tf.layers.dense(tf.layers.flatten(net), z_dim, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            z = tf.nn.relu(tf.layers.batch_normalization(z, training=is_training), name='enc')
            return z


        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape,name="input")
        self.y = tf.placeholder(tf.int32, shape=[None], name="label")
        self.is_training = tf.placeholder(tf.bool,name="is_training")
        # network architecture
        batch_norm_decay = 0.997 if self.config.get('bn_decay')==None else self.config.bn_decay
        output_stride =  self.config.get('output_stride')
        if self.config.model == "resnet":
            with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                net, end_points = resnet_v2.resnet_v2_50(self.x, N_CLASS, is_training=self.is_training, output_stride=output_stride)
                logits = tf.squeeze(end_points["resnet_v2_50/logits"], axis=[1, 2])
            pred = tf.nn.softmax(logits, "pred")
        elif self.config.model == "resnet_101":
            with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                pred, end_points = resnet_v2.resnet_v2_101(self.x, N_CLASS, is_training=self.is_training, output_stride=output_stride)
                logits = tf.squeeze(end_points["resnet_v2_101/logits"], axis=[1, 2])
            pred = tf.nn.softmax(logits)
        elif self.config.model == "resnet_v1_50":
            with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                pred, end_points = resnet_v1.resnet_v1_50(self.x, N_CLASS, is_training=self.is_training, output_stride=output_stride)
                logits = tf.squeeze(end_points["resnet_v1_50/logits"], axis=[1, 2])
            pred = tf.nn.softmax(logits)
        else:
            z = encoder(self.x,z_dim=Z_DIM, is_training=self.is_training)
            logits = tf.layers.dense(z, N_CLASS, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            pred = tf.nn.softmax(logits)

        self.pred = pred
        with tf.name_scope("loss"):

            self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=logits), name='cross_entropy')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(pred, 1, output_type=tf.int32), self.y)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        # 保存bn的m和v。
        # print(bn_moving_vars)
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.config.max_to_keep)

