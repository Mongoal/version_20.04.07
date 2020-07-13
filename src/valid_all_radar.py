# 用来计算两张人脸图像之间的距离矩阵。需要输入的参数：
# 预训练模型 图片1  图片220170512-110547 1.png 2.png

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from src import facenet, radar_io
import matplotlib.pyplot as plt
# import detect_face


train_set,train_indices = radar_io.get_dataset(r'E:\lfw\data3\group28\train')
test_set,test_indices = radar_io.get_dataset(r'E:\lfw\data3\group28\test')
unknown_set,unknown_indices = radar_io.get_dataset(r'E:\lfw\data3\group28\unknown')
num_train_class = len(train_indices)
num_unknown_class = len(unknown_indices)
# plt.figure()
# plt.imshow(images[1,:])
# plt.show()
# print('askhnauisd')

with tf.Graph().as_default():

    with tf.Session() as sess:
        model = 'models/20191022-130933'
        # Load the model
        facenet.load_model(model)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("image_paths:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        batch_size = 1000

        # train_set : Run forward pass to calculate embeddings
        emb_train =np.zeros((len(train_set), 128))
        for i in range(len(train_set)//batch_size+1):
            indices = range(batch_size*i, min(batch_size*i+batch_size,len(train_set)))
            feed_dict = {images_placeholder: train_set[indices], phase_train_placeholder: False}
            emb_train[indices,:] = sess.run(embeddings, feed_dict=feed_dict)

        # test_set : Run forward pass to calculate embeddings
        emb_test =np.zeros((len(test_set), 128))
        for i in range(len(test_set)//batch_size+1):
            indices = range(batch_size*i, min(batch_size*i+batch_size,len(test_set)))
            feed_dict = {images_placeholder: test_set[indices], phase_train_placeholder: False}
            emb_test[indices,:] = sess.run(embeddings, feed_dict=feed_dict)

        # train_set : Run forward pass to calculate embeddings
        emb_unknown =np.zeros((len(unknown_set), 128))
        for i in range(len(unknown_set)//batch_size+1):
            indices = range(batch_size*i, min(batch_size*i+batch_size,len(unknown_set)))
            feed_dict = {images_placeholder: unknown_set[indices], phase_train_placeholder: False}
            emb_unknown[indices,:] = sess.run(embeddings, feed_dict=feed_dict)


'''
 emb center
'''
# train test unknown
emb_center_train = np.zeros((num_train_class,128))
emb_center_test = np.zeros((num_train_class,128))
emb_center_unknown = np.zeros((num_unknown_class,128))
for i, x in enumerate(train_indices):
    emb_center_train[i, :] = np.mean(emb_train[x.indices],0)
    emb_center_train[i, :] /= np.linalg.norm(emb_center_train[i, :])
for i, x in enumerate(test_indices):
    emb_center_test[i, :] = np.mean(emb_test[x.indices],0)
    emb_center_test[i, :] /= np.linalg.norm(emb_center_test[i, :])
for i, x in enumerate(unknown_indices):
    emb_center_unknown[i, :] = np.mean(emb_unknown[x.indices],0)
    emb_center_unknown[i, :] /= np.linalg.norm(emb_center_unknown[i, :])

'''
 distance among emb centers
'''
# among train set emb centers
dist_train_emb_center = np.zeros((num_train_class, num_train_class))
for i,m in enumerate(emb_center_train):
    for j, n in enumerate(emb_center_train):
        dist_train_emb_center[i,j] = np.sqrt(np.sum(np.square(m-n)))
# among unknown set emb centers
dist_unknown_emb_center = np.zeros((num_unknown_class, num_unknown_class))
for i,m in enumerate(emb_center_unknown):
    for j, n in enumerate(emb_center_unknown):
        dist_unknown_emb_center[i,j] = np.sqrt(np.sum(np.square(m-n)))

# between test set emb centers & train set emb centers
dist_test_train_emb_center = np.zeros((num_train_class, num_train_class))
for i,m in enumerate(emb_center_test):
    for j, n in enumerate(emb_center_train):
        dist_test_train_emb_center[i,j] = np.sqrt(np.sum(np.square(m-n)))

# between unknown set emb centers & train set emb centers
dist_unknown_train_emb_center = np.zeros((num_unknown_class, num_train_class))
for i,m in enumerate(emb_center_unknown):
    for j, n in enumerate(emb_center_train):
        dist_unknown_train_emb_center[i,j] = np.sqrt(np.sum(np.square(m-n)))

'''
bias to centers 
'''
# train
bias_train = np.zeros(len(train_set))
mean_bias_train = np.zeros(num_train_class)
std_bias_train = np.zeros(num_train_class)
for i, x in enumerate(train_indices):
    bias_train[x.indices] =  np.sqrt(np.sum(np.square(emb_train[x.indices] - emb_center_train[i, :] ),1))
    mean_bias_train[i] = np.mean(bias_train[x.indices])
    std_bias_train[i] = np.std(bias_train[x.indices])
    a = plt.hist(bias_train[x.indices])
    plt.vlines(dist_train_emb_center[i], 0 ,a[0].max())
    plt.title('bias to center, train '+x.name+'\n'+'mean = '+str( mean_bias_train[i])[:5]+', std = '+str(std_bias_train[i])[:5])
    plt.savefig('bias to center train ' + x.name + '.png')
    plt.show()

# test
bias_test = np.zeros(len(test_set))
mean_bias_test = np.zeros(num_train_class)
std_bias_test = np.zeros(num_train_class)
for i, x in enumerate(test_indices):
    bias_test[x.indices] =  np.sqrt(np.sum(np.square(emb_test[x.indices] - emb_center_test[i, :] ), 1))
    mean_bias_test[i] = np.mean(bias_test[x.indices])
    std_bias_test[i] = np.std(bias_test[x.indices])
    a = plt.hist(bias_test[x.indices])
    plt.vlines(dist_train_emb_center[i], 0 ,a[0].max())
    plt.title('bias to center, test '+x.name+'\n'+'mean = '+str( mean_bias_test[i])[:5]+', std = '+str(std_bias_test[i])[:5])
    plt.savefig('bias to center test ' + x.name + '.png')
    plt.show()
# unknown
bias_unknown = np.zeros(len(unknown_set))
mean_bias_unknown = np.zeros(num_unknown_class)
std_bias_unknown = np.zeros(num_unknown_class)
for i, x in enumerate(unknown_indices):
    bias_unknown[x.indices] =  np.sqrt(np.sum(np.square(emb_unknown[x.indices] - emb_center_unknown[i, :] ),1))
    mean_bias_unknown[i] = np.mean(bias_unknown[x.indices])
    std_bias_unknown[i] = np.std(bias_unknown[x.indices])
    a = plt.hist(bias_unknown[x.indices])
    plt.vlines(dist_unknown_emb_center[i], 0 ,a[0].max())
    plt.title('bias to center, unknown '+x.name+'\n'+'mean = '+str( mean_bias_unknown[i])[:5]+', std = '+str(std_bias_unknown[i])[:5])
    plt.savefig('bias to center unknown '+x.name+'.png')
    plt.show()
plt.subplot
for i, x in enumerate(unknown_indices):
    tx = train_indices[np.argmin(dist_unknown_train_emb_center[i])]
    bias_unknown[x.indices] =  np.sqrt(np.sum(np.square(emb_unknown[x.indices] - emb_center_train[np.argmin(dist_unknown_train_emb_center[i]), :] ),1))
    mean_bias_unknown[i] = np.mean(bias_unknown[x.indices])
    std_bias_unknown[i] = np.std(bias_unknown[x.indices])
    a = plt.hist(bias_unknown[x.indices])
    plt.hist(bias_train[tx.indices])
    plt.title('distance to train center, unknown '+x.name+' & train '+tx.name)
    plt.show()

for x in train_indices:
    for i,h in enumerate(np.random.permutation(len(x))[:3]):
        plt.figure(figsize = (8,4))
        plt.imshow(emb_train[x.indices[h]].reshape(8, 16), cmap=plt.cm.Oranges)
        plt.colorbar()
        plt.title('train '+x.name+' '+str(i))
        plt.savefig('train '+x.name+' '+str(i)+'.png')
        plt.show()
#
# emb_center_train = np.mean(emb_train,1)
# emb_center_test = np.mean(emb_train,1)
#
# # train_set, mean std
# num_class = num_train_class
# train_mean,train_std = np.zeros((num_class,num_class)),np.zeros((num_class,num_class))
# for m in range(num_class):
#     print('train_set %d is %s ' % (m ,train_indices[m].name))
#     indices = train_indices[m].indices
#     dist = []
#     for i in range( min(1000,len(indices))):
#         for j in range(i + 1, min(1000,len(indices))):
#             dist.append(np.sqrt(np.sum(np.square(np.subtract(emb_train[indices[i], :], emb_train[indices[j], :])))))
#     train_mean[m,m] = np.mean(dist)
#     train_std[m,m] = np.std(dist)
# for m in range(num_class):
#     for n in range(num_class):
#         if m != n :
#             dist = []
#             for i in train_indices[m].indices[:min(1000,len(train_indices[m]))]:
#                 for j in train_indices[n].indices[:min(1000,len(train_indices[n]))]:
#                     dist.append(np.sqrt(
#                         np.sum(np.square(np.subtract(emb_train[i, :], emb_train[j, :])))))
#             train_mean[m, n] = np.mean(dist)
#             train_std[m, n] = np.std(dist)
#
#
# # train_test_dist, mean std
# num_class = num_train_class
# train_test_dist_mean,train_test_dist_std = np.zeros((num_class,num_class)),np.zeros((num_class,num_class))
#
# for m in range(num_class):
#     for n in range(num_class):
#         dist = []
#         for i in train_indices[m].indices[:min(1000,len(train_indices[m]))]:
#             for j in test_indices[n].indices[:min(1000,len(test_indices[n]))]:
#                 dist.append(np.sqrt(
#                     np.sum(np.square(np.subtract(emb_train[i, :], emb_test[j, :])))))
#         train_test_dist_mean[m, n] = np.mean(dist)
#         train_test_dist_std[m, n] = np.std(dist)
#
# # unknown_train_dist, mean std
#
# unknown_num_class = len(unknown_indices)
# unknown_train_dist_mean,unknown_train_dist_std = np.zeros((unknown_num_class,num_train_class)),np.zeros((unknown_num_class,num_train_class))
#
# for m in range(unknown_num_class):
#     for n in range(num_train_class):
#         dist = []
#         for i in unknown_indices[m].indices[:min(1000,len(unknown_indices[m]))]:
#             for j in train_indices[n].indices[:min(1000,len(train_indices[n]))]:
#                 dist.append(np.sqrt(
#                     np.sum(np.square(np.subtract(emb_unknown[i, :], emb_train[j, :])))))
#         unknown_train_dist_mean[m, n] = np.mean(dist)
#         unknown_train_dist_std[m, n] = np.std(dist)
#
#
#
#
#

# nrof_images = len(args.csv_paths)
#
# print('Images:')
# for i in range(nrof_images):
#     print('%1d: %s' % (i, args.csv_paths[i]))
# print('')
#
# # Print distance matrix
# print('Distance matrix')
# print('    ', end='')
# for i in range(nrof_images):
#     print('    %1d     ' % i, end='')
# print('')
# for i in range(nrof_images):
#     print('%1d  ' % i, end='')
#     for j in range(nrof_images):
#         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
#         print('  %1.4f  ' % dist, end='')
#     print('')

