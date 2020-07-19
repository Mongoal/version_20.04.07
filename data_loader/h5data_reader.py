# -*- coding: utf-8 -*-
# @Time    : 2020-03-29 22:04
# @Author  : Lin Wanjie
# @File    : h5data_reader.py
import os

import h5py
import numpy as np


# 打开h5 ，打乱，获取batch，
class H5DataReader(object):

    def __init__(self, file, mode='r', data_key='features', label_key='labels', seed=0, train_set_ratio=0.7,
                 seg_set_method='shuffle', txt_path=None):
        '''
        初始化h5读取器，
        :param file: 要读取的文件路径
        :param mode: 'r'
        :param data_key: 数据的key
        :param label_key:  标签的key
        :param seed:  random seed 用于打乱
        :param train_set_ratio: 训练集占比 如果不区分训练集测试集可不设置
        :param seg_set_method: 划分训练集的方法，有'continuous'、TODO 'shuffle'、TODO 'txt' 三种选项
        :param txt_path: TODO 划分训练集的方法=‘txt’ 时，需指定训练集txt文件
        '''
        self._file = h5py.File(file, mode)
        self.data_key = data_key
        self.label_key = label_key
        self._data = self._file[data_key]
        self._labels = self._file[label_key]
        self.length = self._labels.shape[0]
        self.seed = seed
        self.current_id = 0
        np.random.seed(self.seed)
        self.set_idx(train_set_ratio=train_set_ratio, seg_set_method=seg_set_method, txt=txt_path)

    def __str__(self):
        return str(self._file)

    def set_idx(self, train_set_ratio=0.7, seg_set_method='shuffle', txt=None):
        self.shuffle_indices = np.random.permutation(self.length)
        if seg_set_method == 'continuous':
            self.train_indices = np.random.permutation(int(len(self.shuffle_indices)* train_set_ratio))
            self.test_indices = np.arange(int(len(self.shuffle_indices) * train_set_ratio), self.length)
        elif seg_set_method == 'shuffle':
            if (txt is not None and os.path.isfile(txt)):
                self.shuffle_indices = np.loadtxt(txt, dtype=np.int32, delimiter='\n')
            self.train_indices = self.shuffle_indices[:int(len(self.shuffle_indices) * train_set_ratio)]
            self.test_indices = self.shuffle_indices[int(len(self.shuffle_indices) * train_set_ratio):]

            np.savetxt('shuffle_idx.txt', self.shuffle_indices, '%d', '\n')
            np.savetxt('train_idx.txt', self.train_indices, '%d', '\n')
            np.savetxt('test_idx.txt', self.test_indices, '%d', '\n')

        elif seg_set_method == 'txt':
            self.shuffle_indices = np.random.permutation(np.loadtxt(txt, dtype=np.int32, delimiter='\n'))
            self.train_indices = self.shuffle_indices[:int(len(self.shuffle_indices)* train_set_ratio)]
            self.test_indices = self.shuffle_indices[int(len(self.shuffle_indices) * train_set_ratio):]
        elif seg_set_method == 'array':
            self.shuffle_indices = np.random.permutation(txt)
            if train_set_ratio is not None:
                self.train_indices = self.shuffle_indices[:int(len(self.shuffle_indices)* train_set_ratio)]
                self.test_indices = self.shuffle_indices[int(len(self.shuffle_indices) * train_set_ratio):]

    def set_condition_idx(self, condition_keys: list = ['labels', 'fc'],
                          include_conditions: list = None,
                          exclude_conditions: list = None, outfile: str = None, shuffle=False, seg_ratio=None):
        idx = self.get_condition_idx(condition_keys,include_conditions,exclude_conditions,outfile,shuffle)
        self.set_idx(seg_ratio,seg_set_method='array',txt=idx)


    def get_condition_idx(self, condition_keys: list = ['labels', 'fc'],
                          include_conditions: list = None,
                          exclude_conditions: list = None, outfile: str = None, shuffle=False, seg_ratio=None):
        if condition_keys is None or condition_keys == []:
            return []
        # for i in zip(*[[1,2],[3,4]]):
        #     print(i)
        # out:
        # (1, 3)
        # (2, 4)

        included = np.ones(self.length, dtype=np.bool)
        # 如果没有指定include条件（白名单），默认整个数据集
        if include_conditions is not None:
            # 如果有include条件，只留下白名单的编号
            combines = zip(*[self._file[key][:] for key in condition_keys])
            included = np.asarray([e in include_conditions for e in combines])
        # 在白名单里除去黑名单
        if exclude_conditions is not None:
            combines = zip(*[self._file[key][:] for key in condition_keys])
            not_excluded = np.asarray([e not in exclude_conditions for e in combines])
            included = np.logical_and(included, not_excluded)
        idx = np.where(included)[0]
        if shuffle:
            idx = np.random.permutation(idx)
        if seg_ratio is not None:
            idx = np.random.permutation(idx)
            num = int(seg_ratio * self.length)
            idx = (idx[:num], idx[num:])
            if outfile is not None:
                np.savetxt('train_' + outfile, idx[0], fmt='%d', delimiter='\n',
                           header="# condition key: %s, include: %s, exclude: %s" % (
                           condition_keys, include_conditions, exclude_conditions))
                np.savetxt('test_' + outfile, idx[1], fmt='%d', delimiter='\n',
                           header="# condition key: %s, include: %s, exclude: %s" % (
                           condition_keys, include_conditions, exclude_conditions))
        else:
            if outfile is not None:
                np.savetxt(outfile, idx, fmt='%d', delimiter='\n',
                           header="# condition key: %s, include: %s, exclude: %s" % (
                           condition_keys, include_conditions, exclude_conditions))
        return idx

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.shuffle_indices = np.random.permutation(self.length)
        self.current_id = 0

    def get_train_batch(self, batch_size):
        '''
        每次返回，打乱的、不重复的、长度为batch_size的训练集数据和标签的list
        :param batch_size: int
        :return: data_list, labels_list
        '''
        indices = self.train_indices[self.current_id:self.current_id + batch_size]
        data_list = [self._data[idx] for idx in indices]
        labels_list = [self._labels[idx] for idx in indices]
        if (self.current_id + batch_size >= len(self.train_indices)):
            self.current_id = 0
            self.train_indices = np.random.permutation(self.train_indices)
        else:
            self.current_id += batch_size
        return data_list, labels_list

    def get_test_batch(self, batch_size):
        '''
        不打乱，每次返回，长度为batch_size的测试集数据和标签的list
        :param batch_size: int
        :return: data_list, labels_list
        '''
        indices = self.test_indices[self.current_id:self.current_id + batch_size]
        data_list = [self._data[idx] for idx in indices]
        labels_list = [self._labels[idx] for idx in indices]
        if (self.current_id + batch_size >= len(self.test_indices)):
            self.current_id = 0
        else:
            self.current_id += batch_size
        return data_list, labels_list

    def get_shuffle_data(self, batch_size):
        '''
        每次返回，打乱的、不重复的、长度为batch_size的数据和标签的list
        :param batch_size: int
        :return: data_list, labels_list
        '''
        indices = self.shuffle_indices[self.current_id:self.current_id + batch_size]
        data_list = [self._data[idx] for idx in indices]
        labels_list = [self._labels[idx] for idx in indices]
        if (self.current_id + batch_size >= self.length):
            self.current_id = 0
            self.shuffle_indices = np.random.permutation(self.shuffle_indices)
        else:
            self.current_id += batch_size
        return data_list, labels_list

    def shuffle(self):
        self.shuffle_indices = np.random.permutation(self.length)
