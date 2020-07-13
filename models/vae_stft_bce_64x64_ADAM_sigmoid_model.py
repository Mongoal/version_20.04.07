from base.base_model import BaseModel
import tensorflow as tf
from models.cnn_utils import *



class VAEConv2dModel(BaseModel):
    def __init__(self, config):
        super(VAEConv2dModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # 默认参数
        FILTER_SIZE = (5, 5)
        Z_DIM = 256
        STRIDE = (2, 2)
        DEPTHS = [64, 128, 256, 256, 128, 32]
        CHANNELS = 4

        def encoder(input, is_training=False):
            net = conv2d_BN(input, DEPTHS[0], FILTER_SIZE, is_training, stride=STRIDE, name='conv_1',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN(net, DEPTHS[1], FILTER_SIZE, is_training, stride=STRIDE, name='conv_2',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN(net, DEPTHS[2], FILTER_SIZE, is_training, stride=STRIDE, name='conv_3',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            fully_conn = tf.layers.dense(tf.layers.flatten(net), 2048, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            h = tf.nn.relu(tf.layers.batch_normalization(fully_conn, training=is_training), name='enc')
            return h

        def latent_encoder(h, z_dim=256, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)):
            '''
            vae核心

            Args:
                h:
                z_dim:
                kernel_initializer:

            Returns: z和kl_loss

            '''
            gaussian_params = tf.layers.dense(h, 2*z_dim, kernel_initializer=kernel_initializer)
            # The mean parameter is unconstrained
            z_mu = gaussian_params[:, :z_dim]
            # The log(var)
            z_logvar =gaussian_params[:, z_dim:]
            # stddev = exp(z_logvar/2)
            z = tf.add(z_mu , tf.exp(z_logvar/2) * tf.random_normal(shape=tf.shape(z_mu)), name='z')
            kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar)
            print('kl_loss shape: ',kl_loss.shape)
            return z, kl_loss

        def decoder(z, is_training=False ):
            fully_conn = tf.layers.dense(z, 8 * 8 * 256,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            fully_conn_bn = tf.nn.relu(tf.layers.batch_normalization(fully_conn, training=is_training))
            net = tf.reshape(fully_conn_bn, (tf.shape(z)[0], 8, 8, 256))
            net = conv2d_BN_transpose(net, DEPTHS[3], FILTER_SIZE, is_training, stride=STRIDE, name='deconv_1',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN_transpose(net, DEPTHS[4], FILTER_SIZE, is_training, stride=STRIDE, name='deconv_1',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN_transpose(net, DEPTHS[5], FILTER_SIZE, is_training, stride=STRIDE, name='deconv_2',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = tf.layers.conv2d(net, CHANNELS, FILTER_SIZE, padding='same',
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = tf.identity(net, name="recon_unact")
            return net

        '''
        构建模型
        '''
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape,name="input")
        self.is_training = tf.placeholder(tf.bool,name="is_training")
        print(self.x.shape)
        # network architecture
        enc = encoder(self.x, self.is_training)
        z, self.kl_loss = latent_encoder(enc, Z_DIM)
        dec_logits = decoder(z, self.is_training)
        self.decode = tf.sigmoid(dec_logits,name='decode')
        with tf.name_scope("loss"):

            self.recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=dec_logits), name='cross_entropy')
            self.loss = self.recon_loss + self.kl_loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # print(update_ops)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                             global_step=self.global_step_tensor)

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

