import os
import numpy as np
import scipy.signal as Sig
import matplotlib.pyplot as plt
from glob import glob
import struct

import h5py
import time
def make_dataset_h5(directory, pattern='**/*.dat',outpath = 'LTE_dataset_5c_1202.h5'):
    '''
    生成全信号STFT预处理的h5
    :param dir:
    :param film:
    :return:
    '''
    h5f = h5py.File(outpath,'w')

    docs = os.listdir(directory)
    features = []
    for idx,doc in enumerate(docs):

        filepath = directory + '\\' + doc

        files = sorted(glob(filepath + '/' + pattern, recursive=True))

        signal = []
        num_samples_a_class = 0
        n = 0
        for file in files:
            if n >= 10 :
                break
            timer = time.time()
            sig = read_dat(file)

            samples = energy_detect_N_cut(sig)

            num_samples_a_file = len(samples)
            print(file,' | num samples :',num_samples_a_file)
            num_samples_a_class += num_samples_a_file

            for i,sample in enumerate(samples):

                feat_mat = myfft1(sample, nperseg=64, nfft=256, noverlap=52)
                features.append(feat_mat)

                if i % 100 == 99 or i == num_samples_a_file - 1:

                    save_h5(h5f, np.stack(features), 'features')
                    save_h5(h5f, np.ones(len(features), dtype=np.int8) * idx, 'labels')

                    features.clear()
            n+=1
            print(file, ' | times :', time.time()-timer,' s')

        print(idx, ' name:', doc, ' samples:', num_samples_a_class)

    h5f.close()


def read_dat(path):
    '''
    读取二进制_int16_2Channels的数据
    :param path:数据路径
    :return:shape 为[N,2]的信号
    '''
    signal = np.fromfile(path, dtype=np.int16).reshape((-1,2))
    return signal

def energy_detect_N_cut(origin_signal, window_size = 100, gate = 1e3, num_win_per_sample= 30):
    '''
    能量检测，时域切割
    :param origin_signal: numpy ndarray, shape = [Num, 2]，原始时域信号, I、Q两路
    :param window_size:
    :param gate:
    :return: numpy ndarray, shape = [n, window_size * num_win_per_sample, 2], 切割后的时域信号, I、Q两路
    '''

    analyze = origin_signal[:,0]+np.asarray(1j,np.complex64)*origin_signal[:,1]
    analyze = analyze - np.mean(analyze)

    energy_len = int(np.floor(len(analyze) / window_size))
    energy = np.zeros( energy_len)
    for i in range(energy_len):
        head = i * window_size
        tail = i * window_size + window_size
        energy[i] = np.linalg.norm(analyze[head:tail])

    indices = energy > gate
    samples =[]
    i = 0
    while i <= (len(energy) - num_win_per_sample ):
        if ~indices[i]:
            i += 1
            continue
        flag = ~indices[i:i+num_win_per_sample]
        if np.any(flag):
            i += np.where(flag)[0][-1] + 1
        else:
            samples.append(analyze[i*window_size:(i + num_win_per_sample)*window_size])
            i += num_win_per_sample
    return samples

def energy_detect_N_cut_return_dict(origin_signal, window_size = 100, gate = 1e3, num_win_per_sample= 30):
    '''
    能量检测，时域切割
    :param origin_signal: numpy ndarray, shape = [Num, 2]，原始时域信号, I、Q两路
    :param window_size:
    :param gate:
    :return: numpy ndarray, shape = [n, window_size * num_win_per_sample, 2], 切割后的时域信号, I、Q两路
    '''

    analyze = origin_signal[:,0]+np.asarray(1j,np.complex64)*origin_signal[:,1]
    analyze = analyze - np.mean(analyze)

    energy_len = int(np.floor(len(analyze) / window_size))
    energy = np.zeros( energy_len)
    for i in range(energy_len):
        head = i * window_size
        tail = i * window_size + window_size
        energy[i] = np.linalg.norm(analyze[head:tail])

    indices = energy > gate
    samples =[]
    pos = []
    mean_energy= []
    i = 0
    while i <= (len(energy) - num_win_per_sample ):
        if ~indices[i]:
            i += 1
            continue
        flag = ~indices[i:i+num_win_per_sample]
        if np.any(flag):
            i += np.where(flag)[0][-1] + 1
        else:
            mean_energy.append(np.mean(energy[i:i+num_win_per_sample]))
            pos.append(i*window_size)
            samples.append(analyze[i*window_size:(i + num_win_per_sample)*window_size])
            i += num_win_per_sample

    return {'samples':samples, 'position':pos, 'energy':mean_energy}

def myfft1(signal, nperseg=64, nfft=256, noverlap=52):
    _, _, z = Sig.stft(signal, nperseg=64, nfft=256, noverlap=52, return_onesided= False)
    z = np.fft.fftshift(z, 0)
    z = z/np.max(abs(z))
    feature = np.zeros((z.shape[0], z.shape[1], 4), np.float32)
    feature[:,:,0] = z.real
    feature[:,:,1] = z.imag
    feature[:,:,2] = np.log10(abs(z))
    feature[:,:,3] = np.angle(z)
    return feature

def myfft_for_display_1(origin_signal,
                        nperseg=250,
                        nfft=512,
                        noverlap=200,
                        batch_size = 1000000):

    analyze = origin_signal[:,0]+np.asarray(1j,np.complex64)*origin_signal[:,1]
    analyze = analyze - np.mean(analyze)
    l = len(origin_signal)
    features = []
    for s in range(0,l-1,batch_size):
        e = min(s + batch_size, l)
        _, _, z = Sig.stft(analyze[s:e], nperseg=nperseg, nfft=nfft, noverlap=noverlap ,return_onesided= False)
        z = np.fft.fftshift(z, 0)
        feature = 10 * np.log10(abs(z))
        features.append(feature)
    return features

def diplay_1(sig,
             pos,
             predict,
             enengy,
             figname = 'display',
             nperseg=250,
             step = 50,
             nfft=512,
             batch_size = 1000000):
    '''
    分析预测的错误样本分布，能量分布等
    :param sig: 时域信号的path或numpy.ndarray
    :param pos: 样本起始点未知数组
    :param predict: 预测结果数组
    :param enengy: 样本平均能量数组
    :param nperseg:
    :param step:
    :param nfft:
    :param batch_size: 一批对应原始信号上的多少个点
    :return:
    '''
    if type(sig) == 'str':
        figname = sig[:-4]
        sig = read_dat(sig)

    f = myfft_for_display_1(sig,nperseg=nperseg, nfft=nfft, noverlap=nperseg-step, batch_size = batch_size)
    for i in range(len(f)):

        samples_batch_indices = np.asarray(pos) < (i+1)*batch_size & np.asarray(pos) >= i*batch_size

        p = np.asarray(pos)[samples_batch_indices] // step

        e = 10 * np.log10( np.asarray(enengy, np.float32)[samples_batch_indices] )

        c = np.asarray(predict, np.int8)[samples_batch_indices]

        plt.figure(figsize=(180, 5))
        plt.imshow(f[i])
        for j in range(len(p)):
            color = ['red','black','yellow','white','aqua']
            plt.plot([p[j], p[j]+30 ], [e[j], e[j]], color=color[c[j]], linewidth=10 ,label = 'predicted as '+str(c))
        plt.legend()
        plt.savefig(figname+str(i)+'.jpg')


def save_h5(h5f,data,target):
    shape_list=list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0]=None #设置数组的第一个维度是0
        dataset = h5f.create_dataset(target, data=data,maxshape=tuple(shape_list), chunks=True)
        return
    else:
        dataset = h5f[target]
    len_old=dataset.shape[0]
    len_new=len_old+data.shape[0]
    shape_list[0]=len_new
    dataset.resize(tuple(shape_list)) #修改数组的第一个维度
    dataset[len_old:len_new] = data  #存入新的文件

def get_train_valid_indices(number_samples, train_percent = 0.6, random_seed = 666):
    '''

    :param number_samples: int, dataset的样本总数
    :param train_percent:
    :param random_seed:
    :return: 不重复的训练集和验证集索引
    '''
    np.random.seed(random_seed)
    indices = np.random.permutation(number_samples)
    num_train = int(number_samples * train_percent)
    train_indices = indices[:num_train]
    valid_indices = indices[num_train:]
    return train_indices, valid_indices

# path = 'E:\\DATA\\Desktop\\LTE.m\\dat\\split'
# make_dataset_stft(path)


