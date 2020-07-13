

#这个函数整个就是训练的时候所用到的函数集合，用来被其他函数调用的，自己不能执行

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import h5py

PRIMAX = 8e4
PWMAX = 30
RFMAX = 6400
def get_h5dataset(path, num_train_class, phase = 'TRAIN', num_train_sample_per_class = 0.8):
    dataset = []
    num_samples = 0
    label_indices = []
    with h5py.File(path,'r') as f:
        keys = [key for key in f.keys()]

        if phase =='TRAIN':
            for key in keys[:num_train_class]:
                border = int(num_train_sample_per_class * len(f[key]))
                dataset.append(np.array(f[key][:border], np.float32))
                label_indices.append(RadarClass(key, np.arange(num_samples,num_samples+border)))
                num_samples += border
        elif phase == 'TEST':
            for key in keys[:num_train_class]:
                border = int(num_train_sample_per_class * len(f[key]))
                dataset.append(np.array(f[key][border:], np.float32))
                label_indices.append(RadarClass(key, np.arange(num_samples,num_samples+len(f[key])-border)))
                num_samples += len(f[key])-border
        else:
            for key in keys[num_train_class:]:
                dataset.append(np.array(f[key], np.float32))
                label_indices.append(RadarClass(key, np.arange(num_samples, num_samples + len(f[key]) )))
                num_samples += len(f[key])
        dataset = np.concatenate(dataset)
        dataset = dataset.transpose((0,2,3,1))
        return dataset,label_indices

def get_dataset_and_labels(path):
    dataset = []
    labels = []
    # 简单理解就是规范化linux和windows下的路径名
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # 把姓名都放在列表class这个列表里
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        csv_dir = os.path.join(path_exp, class_name)
        indices = read_csv(csv_dir, dataset)
        labels += [i]*len(indices)
    dataset = np.stack(dataset) / [PRIMAX, PWMAX, RFMAX]
    return dataset, np.array(labels,'int32'), i+1


def get_dataset(path):
    label_indices = []
    dataset = []
    # 简单理解就是规范化linux和windows下的路径名
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # 把姓名都放在列表class这个列表里
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        csv_dir = os.path.join(path_exp, class_name)
        indices = read_csv(csv_dir,dataset)
        label_indices.append(RadarClass(class_name, indices))
    dataset = np.stack(dataset)/[PRIMAX,PWMAX,RFMAX]

    return dataset,label_indices

def read_sample_by_path(csv_path_list):
    res = []
    for path in csv_path_list:
        pdw = np.loadtxt(path, 'float32', delimiter=',')
        res.append(input_1d_to_2d(pdw))
    return np.stack(res)/[PRIMAX,PWMAX,RFMAX]

def read_csv(csv_dir, dataset):
    start = len(dataset)
    if os.path.isdir(csv_dir):
        #如果这个姓名下有图像，则读取图像列表
        csvs = os.listdir(csv_dir)
        for csv in csvs:
            csv_path = os.path.join(csv_dir, csv)
            pdw = np.loadtxt(csv_path, 'float32', delimiter=',')
            if (pdw.size!=0):
                dataset.append(input_1d_to_2d(pdw))
    end = len (dataset)
    return list(range(start,end))

def input_1d_to_2d(data,size = 12):
    w = len(data)
    n = np.ceil(size * size / w)
    temp = np.concatenate([data] * int(n))
    temp = temp[:size * size]
    return temp.reshape((size,size,data.shape[1]))

class RadarClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, indices):
        self.name = name
        self.indices = indices

    def __str__(self):
        return self.name + ', ' + str(len(self.indices)) + ' samples'

    def __len__(self):
        return len(self.indices)
