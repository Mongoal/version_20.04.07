from base.base_model import BaseModel
import tensorflow as tf


class AutoEncodingConv2dBNModel(BaseModel):
    def __init__(self, config):
        super(AutoEncodingConv2dBNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool,name="is_training")
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape,name="input")
        print(self.x.shape)
        # network architecture
        self.logits, self.end_points = model(self.x, self.is_training)
        self.decode = tf.tanh(self.logits)
        with tf.name_scope("loss"):
            # self.loss = tf.reduce_mean(tf.squared_difference(self.x, self.decoding), name='mse')
            self.loss = tf.identity(tf.losses.mean_squared_error(self.x, self.logits),'mse' )
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


def conv1d_transpose(input, output_channels, filter_size, rate):
    """

    :param input: 3D （batch，length，channels)
    :param output_channels:
    :param rate:must be 2*n+1, like 1,3,5,7
    :return:
    """
    input_e = tf.expand_dims(input, 2)
    batch = tf.shape(input)[0]
    zero_left = tf.zeros([batch, input.shape[1], (rate - 1) // 2, input.shape[2]], input.dtype)
    zero_right = tf.zeros([batch, input.shape[1], rate // 2, input.shape[2]], input.dtype)
    input_p = tf.reshape(tf.concat([zero_left, input_e, zero_right], 2), [batch, input.shape[1] * rate, input.shape[2]])
    output = tf.layers.conv1d(input_p, output_channels, filter_size, padding='same',
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return output

def conv2d_BN_transpose(input, depth, filter_size, is_training,
              stride=1, padding='same', activation='relu', kernel_initializer=None, name=None):
    '''

    :param input:
    :param depth:
    :param filter_size:
    :param is_training:
    :param stride:
    :param padding:
    :param activation: 'relu','linear' or 'tanh'
    :param kernel_initializer:
    :param name:
    :return:
    '''
    net = tf.layers.conv2d_transpose(input, depth, filter_size,
                           padding=padding, strides=stride, kernel_initializer=kernel_initializer)
    net = tf.layers.batch_normalization(net, training=is_training)
    if activation.upper() == 'RELU':
        net = tf.nn.relu(net, name=name)
    elif activation.upper() == 'TANH':
        net = tf.nn.tanh(net, name=name)
    else:
        if activation.upper() != 'LINEAR':
            raise Exception("activation must be 'relu' or 'tanh'")
    print(net.shape)
    return net

def model(input, is_training=False):
    N_LAYERS = 17
    FILTER_SIZE = (5,5)
    STRIDE = (2,2)

    DEPTHS = [64, 128, 256, 256, 128,32]
    """
    自编码器-全一维卷积 模型
    :param input: 3D Tensor, shape : [batch, length, 1]
    :return: output of this model
    """
    end_points = {}
    net = conv2d_BN(input, DEPTHS[0], FILTER_SIZE, is_training, stride=STRIDE, name='conv_1',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    net = conv2d_BN(net, DEPTHS[1], FILTER_SIZE, is_training, stride=STRIDE, name='conv_2',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    net = conv2d_BN(net, DEPTHS[2], FILTER_SIZE, is_training, stride=STRIDE, name='conv_3',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    flatten = tf.layers.flatten(net)
    fully_conn = tf.layers.dense(flatten, 2048,  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    fully_conn_bn = tf.nn.relu(tf.layers.batch_normalization(fully_conn, training=is_training),name='code')

    end_points["code"] = fully_conn_bn

    fully_conn = tf.layers.dense(flatten, 8*8*256,  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully_conn_bn = tf.nn.relu(tf.layers.batch_normalization(fully_conn, training=is_training))
    net = tf.reshape(fully_conn_bn,(tf.shape(input)[0],8,8,256))

    net = conv2d_BN_transpose(net, DEPTHS[3], FILTER_SIZE, is_training, stride=STRIDE, name='deconv_1',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net = conv2d_BN_transpose(net, DEPTHS[4], FILTER_SIZE, is_training, stride=STRIDE, name='deconv_1',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net = conv2d_BN_transpose(net, DEPTHS[5], FILTER_SIZE, is_training, stride=STRIDE, name='deconv_2',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net = tf.layers.conv2d(net, input.shape[-1], FILTER_SIZE, padding='same',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.identity(net,name="recon_unact")
    end_points["recon_unact"] = net
    return net, end_points

def conv2d_BN(input, depth, filter_size, is_training,
              stride=1, padding='same', activation='relu', kernel_initializer=None, name=None):
    '''

    :param input:
    :param depth:
    :param filter_size:
    :param is_training:
    :param stride:
    :param padding:
    :param activation: 'relu','linear' or 'tanh'
    :param kernel_initializer:
    :param name:
    :return:
    '''
    net = tf.layers.conv2d(input, depth, filter_size,
                           padding=padding, strides=stride, kernel_initializer=kernel_initializer)
    net = tf.layers.batch_normalization(net, training=is_training)
    if activation.upper() == 'RELU':
        net = tf.nn.relu(net, name=name)
    elif activation.upper() == 'TANH':
        net = tf.nn.tanh(net, name=name)
    else:
        if activation.upper() != 'LINEAR':
            raise Exception("activation must be 'relu' or 'tanh'")

    print(net.shape)
    return net
