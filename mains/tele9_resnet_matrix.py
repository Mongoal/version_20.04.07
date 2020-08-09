from __future__ import absolute_import, division, print_function
import tensorflow as tf
import random
import sys
sys.path.append('..')
# import resnet101 as resnet
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
#from models import densenet as densenet
import tensorflow.contrib.slim as slim
#from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import os
from utils.prepare_v03 import myfft2
from data_loader.h5data_reader import H5DataReader
os.environ['CUDA_VISIBLE_DEVICES'] = '3'    # 使用 GPU 0，1

# 打乱数据集顺序
def shuffle_set(x, label):
    row = list(range(len(label)))
    random.shuffle(row)
    x = x[row]
    label = label[row]
    return x, label

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#文件存放路径

dir_path = 'dataset_signal_5000_new.h5'
#dir_path1 = 'G://data 9（通信电台）//电台数据预处理//'

#数据长度
sig_len = 10000
#训练集数据量比测试集数据量
train_test_rate = 4
#设置训练模式
mode = 'f'
acc_rate = 0.90  # 预设准确率
batch_size = 20
LR = 1e-4
labels = ['dt1', 'dt2', 'dt3', 'dt4', 'dt5', 'dt6', 'dt7', 'dt8', 'dt9']
tick_marks = np.array(range(len(labels))) + 0.5


row = 2000

data = H5DataReader(dir_path,'r','signals','labels')


kind_num = 9
shape1 = 128
shape2 = 128

x = tf.placeholder(tf.float32, shape=(None, shape1, shape2, 4), name = 'x_features')
y_ = tf.placeholder(tf.int64, shape=[None, kind_num], name = 'y_')
keep_prob = tf.placeholder("float", name = 'keep_prob')
training = tf.placeholder(tf.bool, name='training')

n_batch = len(data.train_indices) // batch_size

# prelogits, end_points = network_incep.inference(x, keep_prob,
#                                           phase_train=training , bottleneck_layer_size=128,
#                                           weight_decay=0.0)

#logits = densenet.densenet_inference(x, training, keep_prob)

# prelogits, end_points = resnet.resnet_v1_101(x , is_training=training)
prelogits, end_points = resnet_v2.resnet_v2_101(x , is_training=training)
prelogits = tf.layers.flatten(prelogits)
#logits = densenet.densenet_inference(x, training, keep_prob)

logits = slim.fully_connected(prelogits, kind_num, activation_fn=None,
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                              # weights_regularizer=slim.l2_regularizer(0.0),
                              scope='Logits', reuse=False)

#logits = tf.identity(logits, 'logits')
y_conv = tf.nn.softmax(logits, name='Softmax')

cm = tf.confusion_matrix(tf.argmax(y_,1), tf.argmax(y_conv,1), num_classes=8)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.argmax(y_, 1), logits=logits, name='cross_entropy_per_example')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

opt = tf.train.AdamOptimizer(LR)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = opt.minimize(total_loss)


# 初始化变量
init = tf.global_variables_initializer()

# 首先分别在训练值y_conv以及实际标签值y_的第一个轴向取最大值，比较是否相等
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 对correct_prediction值进行浮点化转换，然后求均值，得到精度。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name = 'acc')

training_epochs = 200
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_los_tol = []
    train_acc_tol = []
    test_los_tol = []
    test_acc_tol = []
    epoch_axis = np.arange(training_epochs)
    epoch_axis = epoch_axis + 1
    for epoch in range(training_epochs):


        for batch in range(n_batch):
            train_data, train_labels = data.get_train_batch(batch_size)
            test_data, test_labels = data.get_test_batch(batch_size)

            x_stft = [myfft2(x) for x in train_data]
            sess.run(train_step, feed_dict={x: x_stft, y_: train_labels, keep_prob: 0.5 ,training: True})

        train_data= data._data[data.train_indices[:200]]
        train_labels= data._labels[data.train_indices[:200]]
        train_data_stft = [myfft2(x) for x in train_data]
        train_loss, train_acc, train_cm = sess.run([total_loss, accuracy, cm],
                                         feed_dict={x: train_data_stft,y_: train_labels, keep_prob: 1.0, training: False})
        train_acc_tol.append(train_acc)
        train_los_tol.append(train_loss)


        test_data= data._data[data.test_indices[:200]]
        test_labels= data._labels[data.test_indices[:200]]
        test_data_stft = [myfft2(x) for x in test_data]

        test_loss, test_acc, test_cm = sess.run([total_loss, accuracy, cm],
                                       feed_dict={x: test_data_stft , y_: test_labels, keep_prob: 1.0,training: False})
        test_acc_tol.append(test_acc)
        test_los_tol.append(test_loss)



        if (epoch+1) % 5 == 0:
            np.set_printoptions(precision=2)
            cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(12, 8), dpi=120)
            ind_array = np.arange(len(labels))
            x_gra, y_gra = np.meshgrid(ind_array, ind_array)
            for x_val, y_val in zip(x_gra.flatten(), y_gra.flatten()):
                c = cm_normalized[y_val][x_val]
                if c > 0.01:
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
            plt.gca().set_xticks(tick_marks, minor=True)
            plt.gca().set_yticks(tick_marks, minor=True)
            plt.gca().xaxis.set_ticks_position('none')
            plt.gca().yaxis.set_ticks_position('none')
            plt.grid(True, which='minor', linestyle='-')
            plt.gcf().subplots_adjust(bottom=0.15)
            plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
            plt.savefig('test_resnetv2_Epoch'+str(epoch+1)+'.png',format='png')
            plt.close()
        if epoch % 1 == 0:
            print("tele9_"+",train/test_rate:"+str(train_test_rate) + ",Epoch" + str(epoch) + ",Train loss = " + str(train_loss) + ",Train Acc = " + str(train_acc) + ",")
            print("tele9_"+",train/test_rate:"+str(train_test_rate) + ",Epoch" + str(epoch) + ",Test loss = " + str(test_loss) + ",Test Acc = " + str(test_acc))



       #  if (epoch+1) % 10 == 0 or epoch == 0:
       #      np.set_printoptions(precision=2)
       #      cm_normalized_train = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]
       #      plt.figure(figsize=(12, 8), dpi=120)
       #      ind_array = np.arange(len(labels))
       #      x_gra, y_gra = np.meshgrid(ind_array, ind_array)
       #      for x_val, y_val in zip(x_gra.flatten(), y_gra.flatten()):
       #          c = cm_normalized_train[y_val][x_val]
       #          if c > 0.01:
       #              plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=20, va='center', ha='center')
       #      plt.gca().set_xticks(tick_marks, minor=True)
       #      plt.gca().set_yticks(tick_marks, minor=True)
       #      plt.gca().xaxis.set_ticks_position('none')
       #      plt.gca().yaxis.set_ticks_position('none')
       #      plt.grid(True, which='minor', linestyle='-')
       #      plt.gcf().subplots_adjust(bottom=0.15)
       #      plot_confusion_matrix(cm_normalized_train, title='Normalized confusion matrix')
       # #     plt.savefig('D:/csgtsb_for_lwj_and_yzl/报告图片/train_mat_train2_test3_310_epoch'+str(epoch+1)+'.png', format='png')
       #      plt.close()
       #
       #      np.set_printoptions(precision=2)
       #      cm_normalized_test = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
       #      plt.figure(figsize=(12, 8), dpi=120)
       #      ind_array = np.arange(len(labels))
       #      x_gra, y_gra = np.meshgrid(ind_array, ind_array)
       #      for x_val, y_val in zip(x_gra.flatten(), y_gra.flatten()):
       #          c = cm_normalized_test[y_val][x_val]
       #          if c > 0.01:
       #              plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=20, va='center', ha='center')
       #      plt.gca().set_xticks(tick_marks, minor=True)
       #      plt.gca().set_yticks(tick_marks, minor=True)
       #      plt.gca().xaxis.set_ticks_position('none')
       #      plt.gca().yaxis.set_ticks_position('none')
       #      plt.grid(True, which='minor', linestyle='-')
       #      plt.gcf().subplots_adjust(bottom=0.15)
       #      plot_confusion_matrix(cm_normalized_test, title='Normalized confusion matrix')
       #      plt.savefig('D:/csgtsb_for_lwj_and_yzl/报告图片/test_mat_train2test2_255_epoch'+str(epoch+1)+'.png', format='png')
       #      plt.close()
  #   plt.plot(epoch_axis,train_los_tol)
  #   plt.plot(epoch_axis,test_los_tol)
  #   plt.xlabel("Epoch")
  #   plt.ylabel("Loss")
  #   plt.title("Variation of Loss during training about group2-255M")
  #   plt.legend(['train','test'])  # 添加图例
  #  # plt.savefig('D:/csgtsb_for_lwj_and_yzl/报告图片/loss_train2test2_255_2.png', format='png')
  #   plt.close()
  #
  #   plt.plot(epoch_axis,train_acc_tol)
  #   plt.plot(epoch_axis,test_acc_tol)
  #   plt.xlabel("Epoch")
  #   plt.ylabel("Acc")
  #   plt.title("Variation of Acc during training about group2-255M")
  #   plt.legend(['train','test'])  # 添加图例
  # #  plt.savefig('D:/csgtsb_for_lwj_and_yzl/报告图片/acc_train2test2_255_2.png', format='png')
  #   plt.close()

   # saver.save(sess=sess, save_path='./model/classification.ckpt')


