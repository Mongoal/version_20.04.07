from base.base_model import BaseModel
import tensorflow as tf


class AutoEncodingConv1dBNModel(BaseModel):
    def __init__(self, config):
        super(AutoEncodingConv1dBNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool,name="is_training")
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape,name="input")
        print(self.x.shape)
        # network architecture
        self.decode, self.end_points = model(self.x, self.is_training)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.x, self.decode), name='mse')

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
    zeros = tf.zeros([batch, input.shape[1], (rate - 1) // 2, input.shape[2]], input.dtype)
    input_p = tf.reshape(tf.concat([zeros, input_e, zeros], 2), [batch, input.shape[1] * rate, input.shape[2]])
    output = tf.layers.conv1d(input_p, output_channels, filter_size, padding='same',
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return output

def conv1d_BN_transpose(input, output_channels, filter_size, rate ,is_training, name=None):
    """

    :param input: 3D （batch，length，channels)
    :param output_channels:
    :param rate:must be 2*n+1, like 1,3,5,7
    :return:
    """
    input_e = tf.expand_dims(input, 2)
    batch = tf.shape(input)[0]
    zeros_left = tf.zeros([batch, input.shape[1], (rate - 1) // 2, input.shape[2]], input.dtype)
    zeros_right = tf.zeros([batch, input.shape[1], rate // 2, input.shape[2]], input.dtype)
    input_p = tf.reshape(tf.concat([zeros_left, input_e, zeros_right], 2), [batch, input.shape[1] * rate, input.shape[2]])
    output = conv1d_BN(input_p, output_channels, filter_size, is_training, name=name, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    return output

def model(input, is_training=False):
    N_LAYERS = 17
    FILTER_SIZE = 5
    STRIDE = 3

    DEPTHS = [32, 48, 72, 108, 162]
    """
    自编码器-全一维卷积 模型
    :param input: 3D Tensor, shape : [batch, length, 1]
    :return: output of this model
    """
    end_points = {}
    net = conv1d_BN(input, DEPTHS[0], FILTER_SIZE, is_training, stride=STRIDE, name='conv_1',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = conv1d_BN(net, DEPTHS[0], 3, is_training, stride=2, name='conv_2',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = conv1d_BN(net, DEPTHS[1], FILTER_SIZE, is_training, stride=STRIDE, name='conv_3',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = conv1d_BN(net, DEPTHS[2], FILTER_SIZE, is_training, stride=STRIDE, name='conv_4',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = conv1d_BN(net, DEPTHS[2], 3, is_training, stride=2, name='conv_5',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = conv1d_BN(net, DEPTHS[3], FILTER_SIZE, is_training, stride=STRIDE, name='conv_6',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = conv1d_BN(net, DEPTHS[4], 3, is_training, stride=2, name='conv_7',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = tf.identity(net,name="code")
    end_points["code"] = net

    net = conv1d_BN_transpose(net, DEPTHS[4], 3, 2, is_training)
    net = conv1d_BN_transpose(net, DEPTHS[3], FILTER_SIZE, STRIDE, is_training)
    net = conv1d_BN_transpose(net, DEPTHS[2], 3, 2, is_training)
    net = conv1d_BN_transpose(net, DEPTHS[2], FILTER_SIZE, STRIDE, is_training)
    net = conv1d_BN_transpose(net, DEPTHS[1], FILTER_SIZE, STRIDE, is_training)
    net = conv1d_BN_transpose(net, DEPTHS[0], 3, 2, is_training)
    net = conv1d_BN_transpose(net, DEPTHS[0], FILTER_SIZE, STRIDE, is_training)
    net = conv1d_BN(net, input.shape[-1], FILTER_SIZE, is_training,  activation='tanh',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    net = tf.identity(net,name="decode")
    end_points["decode"] = net
    return net, end_points

def conv1d_BN(input, depth, filter_size, is_training,
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
    net = tf.layers.conv1d(input, depth, filter_size,
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
