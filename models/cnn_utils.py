import tensorflow as tf
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