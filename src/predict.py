import random
import tensorflow as tf
import numpy as np
import prepare
import load_model
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'

# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_string('data_path', 'lte_dx_5w_25.dat', "测试信号.dat文件路径")
tf.app.flags.DEFINE_integer('start', None, "待测区间起点")
tf.app.flags.DEFINE_integer('end', None, "待测区间终点")

FLAGS = tf.app.flags.FLAGS


# 必须带参数，否则：'TypeError: main() takes no arguments (1 given)';   main的参数名随意定义，无要求
def main(_):
    print(FLAGS.data_path)
    print(FLAGS.start)
    print(FLAGS.end)
    predict(FLAGS.data_path, FLAGS.start, FLAGS.end)

def predict(data_path, start =None, end =None):

    model = 'model/20191203-000349'
    signal = prepare.read_dat(data_path)[start:end]
    print('data_len: ', len(signal))
    print('energy segmentation ... ')
    ti = time.time()
    seg_signal = prepare.energy_detect_N_cut(signal, gate = 1e3)
    num_samples = len(seg_signal)
    to = time.time()
    print('segmentation done! num samples: %d ,times : %3.4f s' % (num_samples, (to-ti)) )

    soft = np.zeros((num_samples,5), np.float32)

    batch_size = 100
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Load the model
            load_model.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("images:0")
            softmax = tf.get_default_graph().get_tensor_by_name("Softmax:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            steps = int(np.ceil(num_samples/batch_size))

            print('predicting ... ')
            ti = time.time()

            for batch_number in range(steps):
                batch_idx = slice(batch_number * batch_size,
                                  min(batch_number * batch_size + batch_size, num_samples))
                features = np.stack([prepare.myfft1(i) for i in seg_signal[batch_idx]])
                feed_dict = {phase_train_placeholder: False,
                             images_placeholder: features }

                soft[batch_idx] = sess.run(softmax, feed_dict=feed_dict)

            predict = np.argmax(soft, 1).astype(np.int8)

            to = time.time()
            print('done! , times : %3.4f s' % (to - ti))
    print(data_path)
    print('number for each predicted class： ', [np.sum(predict == i) for i in range(5)])

    return predict

if __name__ == '__main__':
    tf.app.run()
