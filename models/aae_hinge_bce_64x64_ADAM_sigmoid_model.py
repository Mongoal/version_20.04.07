from base.base_model import BaseModel
import tensorflow as tf
from models.cnn_utils import *



class AAEConv2dModel(BaseModel):
    def __init__(self, config):
        super(AAEConv2dModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # 默认参数
        FILTER_SIZE = (5, 5)
        Z_DIM = self.config.zdim
        self.Z_DIM = Z_DIM
        STRIDE = (2, 2)
        DEPTHS = [64, 128, 256, 256, 128, 32]
        CHANNELS = 4

        def encoder(input, z_dim=100, is_training=False):
            net = conv2d_BN(input, DEPTHS[0], FILTER_SIZE, is_training, stride=STRIDE, name='conv_1',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN(net, DEPTHS[1], FILTER_SIZE, is_training, stride=STRIDE, name='conv_2',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = conv2d_BN(net, DEPTHS[2], FILTER_SIZE, is_training, stride=STRIDE, name='conv_3',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            h = tf.layers.dense(tf.layers.flatten(net), 1024, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            h = tf.nn.relu(tf.layers.batch_normalization(h, training=is_training), name='enc')
            z = tf.layers.dense(h, z_dim,  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            z = tf.nn.relu(tf.layers.batch_normalization(z, training=is_training), name='z')
            return z

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

        def discriminator(z):
            net = tf.layers.dense(z, 128,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = tf.layers.dense(net, 1,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            d = tf.nn.tanh(net, name='d')
            return d
        '''
        构建模型
        '''
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape,name="input")
        self.z = tf.placeholder(tf.float32, shape=[None,Z_DIM])
        self.is_training = tf.placeholder(tf.bool,name="is_training")
        # network architecture
        with tf.variable_scope("enc"):
            z = encoder(self.x, z_dim=Z_DIM, is_training=self.is_training)
        with tf.variable_scope("dec"):
            dec_logits = decoder(z, self.is_training)
            self.decode = tf.nn.sigmoid(dec_logits,name='decode')
        with tf.variable_scope("dis"):
            D = discriminator(tf.concat([z,self.z],axis=0))
        vars_enc = tf.trainable_variables("enc")
        vars_dec = tf.trainable_variables("dec")
        vars_dis = tf.trainable_variables("dis")
        # z ->fake, self.z ->real
        self.D_real = D[:tf.shape(z)[0]]
        self.D_fake = D[tf.shape(z)[0]:]

        with tf.name_scope("loss"):
            self.recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=dec_logits), name='cross_entropy')
            self.D_loss = tf.reduce_mean(tf.nn.relu(1.0 - self.D_real)) +  tf.reduce_mean(tf.nn.relu(1. + self.D_fake))
            self.G_loss = - tf.reduce_mean(self.D_fake)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies(update_ops):
                self.AE_solver = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.recon_loss, var_list=vars_enc + vars_dec,
                                                                                            global_step=self.global_step_tensor)
                self.D_solver = tf.train.AdamOptimizer(self.config.learning_rate*2).minimize(self.D_loss, var_list=vars_dis)
                self.G_solver = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.G_loss, var_list=vars_enc)

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

