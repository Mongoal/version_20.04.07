# -*- coding: utf-8 -*-
# @Time    : 2020-03-27 16:52
# @Author  : Lin Wanjie
# @File    : prepare_v02.py


#  -*- coding: utf-8 -*-
#  @Time    : ${DATE} ${TIME}
#  @Author  : Lin Wanjie
#  @File    : ${NAME}.py


import os
import numpy as np
import scipy.signal as Sig
import matplotlib.pyplot as plt
from glob import glob
import struct

import h5py
import time
from skimage.transform import resize
def read_dat(path):
    '''
    读取二进制_int16_2Channels的数据
    :param path:数据路径
    :return:shape 为[N,2]的信号
    '''
    signal = np.fromfile(path, dtype=np.int16).reshape((-1, 2))
    return signal

def signal_regulation(cut_origin_signal):
    return cut_origin_signal/np.max(np.abs(cut_origin_signal))/2 + 0.5
def signal_regulation_old(cut_origin_signal):
    return cut_origin_signal/np.max(np.abs(cut_origin_signal))

def convert_dataset_from_yzl(directory='/media/ubuntu/90679409-852b-4084-81e3-5de20cfa3035/yzl/tele9/tele9_part',  pattern='**/*.h5', outpath ="dataset_signal_10000_yzl.h5"):
    '''
    yzl目录---201601---xxxFc225xx.mat
           |        |-xxxFc450xx.mat
           |-201602---xxx.mat
     xxx.mat格式 h5. key:sig_valid

    :param directory:yzl mat 根目录，目录下只能有N个文件夹，N为类别数，每个文件夹存放不同类别的dat
    :param film:
    :return:
    '''
    h5f = h5py.File(outpath, 'w')
    docs = os.listdir(directory)
    for label, doc in enumerate(docs):

        filepath = directory + '/' + doc
        files = sorted(glob(filepath + '/' + pattern, recursive=True))

        num_samples_a_class = 0
        n = 0
        for file in files:
            # 计时
            timer = time.time()

            # 读取
            sig = h5py.File(file,'r')['sig_valid'][:]
            samples = np.zeros([len(sig), 2], np.int16)
            for i in range(len(sig)):
                samples[i,0] = int(sig[i,0][0])
                samples[i,1] = int(sig[i,0][1])

            samples = samples.reshape([-1,10000,2])
            print(samples.shape,samples.dtype)
            # append_data_to_h5(h5f, doc, 'label_names')
            append_data_to_h5(h5f, np.stack(samples), 'signals')
            append_data_to_h5(h5f, np.ones(len(samples), dtype=np.int8) * label, 'labels')
            str_idx=file.find('Fc')
            fc = int(file[str_idx+2:str_idx+5])
            append_data_to_h5(h5f, np.ones(len(samples), dtype=np.int32) * fc, 'fc')

            # 统计代码
            num_samples_a_file = len(samples)
            print(file, ' | num samples :', num_samples_a_file)
            num_samples_a_class += num_samples_a_file
            n += 1
            print(file, ' | times :', time.time() - timer, ' s')

        print(label, ' name:', doc, ' samples:', num_samples_a_class)

    h5f.close()

def myfft1_norm(stft):
    stft[:, :, 0] =    stft[:, :, 0]/2 + 0.5
    stft[:, :, 1] = stft[:, :, 1]/2 + 0.5
    stft[:,:,2] =  (stft[:,:,2]+10)/10
    stft[:,:,3] = stft[:,:,3] /2/np.pi + 0.5
    return stft

def signal_normalization(cut_origin_signal):
    power = np.mean(np.square(cut_origin_signal)) * 2
    return cut_origin_signal/np.sqrt(power)

def energy_detect(origin_signal , window_size = 100 ):
    analyze = origin_signal[:, 0] + np.asarray(1j, np.complex64) * origin_signal[:, 1]
    analyze = analyze - np.mean(analyze)
    print(np.mean(analyze))

    energy_len = int(np.floor(len(analyze) / window_size))
    energy = np.zeros(energy_len)
    for i in range(energy_len):
        head = i * window_size
        tail = i * window_size + window_size
        # 能量块的能量定义为， 能量块内信号幅度平方和的开方根，若gate 为1000 ，则对应平均幅度为sqrt(gate**2 /100) = gate /10 = 100
        energy[i] = np.linalg.norm(analyze[head:tail])/np.sqrt(window_size)
    return  energy

def energy_detect_N_cut(origin_signal, window_size=100, gate=1e3, num_win_per_sample=30):
    '''
    能量检测，时域切割
    :param origin_signal: numpy ndarray, shape = [Num, 2]，原始时域信号, I、Q两路
    :param window_size:
    :param gate:
    :return: numpy ndarray, shape = [n, window_size * num_win_per_sample, 2], 切割后的时域信号, I、Q两路
    '''

    analyze = origin_signal[:, 0] + np.asarray(1j, np.complex64) * origin_signal[:, 1]
    analyze = analyze - np.mean(analyze)

    energy_len = int(np.floor(len(analyze) / window_size))
    energy = np.zeros(energy_len)
    for i in range(energy_len):
        head = i * window_size
        tail = i * window_size + window_size
        # 能量块的能量定义为， 能量块内信号幅度平方和的开方根，若gate 为1000 ，则对应平均幅度为sqrt(gate**2 /100) = gate /10 = 100
        energy[i] = np.linalg.norm(analyze[head:tail])

    satisfy_bool_indices = energy > gate
    samples = []
    i = 0
    # 遍历所有能量块
    while i <= (len(energy) - num_win_per_sample):

        # 查询 连续 30 个能量块是否都满足能量限制
        # flag :长度为30的boolean数组 ，若元素e为 true ,则元素e不满足条件
        flag = ~satisfy_bool_indices[i:i + num_win_per_sample]
        # 如果有任意元素不满足，找到最后一个不满足的元素，从那开始
        if np.any(flag):
            i += np.where(flag)[0][-1] + 1
        # 反之，连续30个能量块都满足能量条件，将它们作为一个整体放入样本集合
        else:
            samples.append(analyze[i * window_size:(i + num_win_per_sample) * window_size])
            i += num_win_per_sample
    return samples


def energy_detect_N_cut_origin(origin_signal, window_size=100, gate=1e2, num_win_per_sample=30, drop_abnormal_mean= False):
    '''
    能量检测，时域切割
    :param origin_signal: numpy ndarray, shape = [Num, 2]，原始时域信号, I、Q两路
    :param window_size:
    :param gate:
    :return: numpy ndarray, shape = [n, window_size * num_win_per_sample, 2], 切割后的时域信号, I、Q两路
    '''

    analyze = origin_signal[:, 0] + np.asarray(1j, np.complex64) * origin_signal[:, 1]
    analyze = analyze - np.mean(analyze)

    energy_len = int(np.floor(len(analyze) / window_size))
    energy = np.zeros(energy_len)
    mean = np.zeros(energy_len)
    for i in range(energy_len):
        head = i * window_size
        tail = i * window_size + window_size
        # 能量块的能量定义为， 能量块内信号幅度平方和的开方根，若gate 为1000 ，则对应平均幅度为sqrt(gate**2 /100) = gate /10 = 100
        energy[i] = np.linalg.norm(analyze[head:tail])/np.sqrt(window_size)
        mean[i] = np.mean(origin_signal[head:tail, 0])

    satisfy_bool_indices = np.logical_and(energy > gate , mean < 0.5 * gate)
    samples = []
    i = 0
    # 遍历所有能量块
    while i <= (len(energy) - num_win_per_sample):

        # 查询 连续 30 个能量块是否都满足能量限制
        # flag :长度为30的boolean数组 ，若元素e为 true ,则元素e不满足条件
        flag = ~satisfy_bool_indices[i:i + num_win_per_sample]
        # 如果有任意元素不满足，找到最后一个不满足的元素，从那开始
        if np.any(flag):
            i += np.where(flag)[0][-1] + 1
        # 反之，连续30个能量块都满足能量条件，将它们作为一个整体放入样本集合
        else:
            samples.append(origin_signal[i * window_size:(i + num_win_per_sample) * window_size, :])
            i += num_win_per_sample
    return samples

def make_dataset_fc(directory, pattern='**/*.dat', outpath='dataset_fc.h5'):
    h5f = h5py.File(outpath, 'w')
    docs = os.listdir(directory)
    features = []
    label = 0
    for  doc in docs:

        filepath = directory + '/' + doc
        files = sorted(glob(filepath + '/' + pattern, recursive=True))

        num_samples_a_class = 0
        n = 0
        for file in files:
            if n >= 10:
                break
            # 计时
            timer = time.time()
            str_idx=file.find('Fc')
            fc = int(file[str_idx+2:str_idx+5])
            print(file)
            print(fc)
            sig = read_dat(file)
            # 30 *108 = 3240样本长度，
            samples = energy_detect_N_cut(sig, 108, 3e3)[:10000]
            append_data_to_h5(h5f, np.stack(samples), 'signals')

            # 统计代码
            num_samples_a_file = len(samples)
            for i, sample in enumerate(samples):
                feat_mat = myfft2(sample)
                features.append(feat_mat)

                if i % 100 == 99 or i == num_samples_a_file - 1:
                    append_data_to_h5(h5f, np.stack(features), 'features')
                    append_data_to_h5(h5f, np.ones(len(features), dtype=np.int8) * label, 'labels')
                    append_data_to_h5(h5f, np.ones(len(features), dtype=np.int32) * fc, 'fc')
                    features.clear()



            print(file, ' | num samples :', num_samples_a_file)
            num_samples_a_class += num_samples_a_file
            n += 1
            print(file, ' | times :', time.time() - timer, ' s')

        if len(files) > 0 :
            label += 1
        print(label, ' name:', doc, ' samples:', num_samples_a_class)

    h5f.close()

def make_dataset_signal_fc(directory, pattern='**/*.dat', drop_abnormal_mean= False, outpath='dataset_signal_10000_fc.h5'):
    h5f = h5py.File(outpath, 'w')
    docs = os.listdir(directory)
    label = 0
    for  doc in docs:

        filepath = directory + '/' + doc
        files = sorted(glob(filepath + '/' + pattern, recursive=True))

        num_samples_a_class = 0
        n = 0
        for file in files:
            if n >= 10:
                break
            # 计时
            timer = time.time()
            str_idx=file.find('Fc')
            fc = int(file[str_idx+2:str_idx+5])
            print(file)
            print(fc)
            sig = read_dat(file)
            # 30 *108 = 3240样本长度，
            samples = energy_detect_N_cut_origin(sig, 500, 300, 10, drop_abnormal_mean)[:2000]
            append_data_to_h5(h5f, np.stack(samples), 'signals')
           # append_data_to_h5(h5f, np.stack(features), 'features')
            append_data_to_h5(h5f, np.ones(len(samples), dtype=np.int8) * label, 'labels')
            append_data_to_h5(h5f, np.ones(len(samples), dtype=np.int32) * fc, 'fc')
            # 统计代码
            num_samples_a_file = len(samples)
            print(file, ' | num samples :', num_samples_a_file)
            num_samples_a_class += num_samples_a_file
            n += 1
            print(file, ' | times :', time.time() - timer, ' s')

        if len(files) > 0 :
            label += 1
        print(label, ' name:', doc, ' samples:', num_samples_a_class)

    h5f.close()


def make_dataset_cut_origin_signal(directory, pattern='**/*.dat', outpath='LTE_origin_3240_dataset_5c_10s_1202.h5'):
    '''
    生成全信号经能量过滤、裁剪后的h5
    :param dir:信号根目录，目录下只能有N个文件夹，N为类别数，每个文件夹存放不同类别的dat
    :param film:
    :return:
    '''
    h5f = h5py.File(outpath, 'w')

    docs = os.listdir(directory)
    for label, doc in enumerate(docs):

        filepath = directory + '/' + doc
        files = sorted(glob(filepath + '/' + pattern, recursive=True))

        num_samples_a_class = 0
        n = 0
        for file in files:
            if n >= 10:
                break
            # 计时
            timer = time.time()

            sig = read_dat(file)
            # 30 *108 = 3240样本长度，
            samples = energy_detect_N_cut_origin(sig, 108, 2e3)

            # append_data_to_h5(h5f, doc, 'label_names')
            append_data_to_h5(h5f, np.stack(samples), 'signals')
            append_data_to_h5(h5f, np.ones(len(samples), dtype=np.int8) * label, 'labels')

            # 统计代码
            num_samples_a_file = len(samples)
            print(file, ' | num samples :', num_samples_a_file)
            num_samples_a_class += num_samples_a_file
            n += 1
            print(file, ' | times :', time.time() - timer, ' s')

        print(label, ' name:', doc, ' samples:', num_samples_a_class)

    h5f.close()

def get_h5_condition_idx(h5filepath, condition_keys: list = ['labels', 'fc'],
                      include_conditions: list = [(0, 225), (1, 225), (2, 380)],
                      exclude_conditions: list = [(1, 225)], outfile: str = None, shuffle=False, seg_ratio=None):
    if condition_keys is None or condition_keys == []:
        return []
    # for i in zip(*[[1,2],[3,4]]):
    #     print(i)
    # out:
    # (1, 3)
    # (2, 4)
    h5f = h5py.File(h5filepath,'r')
    length = h5f[condition_keys[0]].shape[0]
    included = np.ones(length, dtype=np.bool)
    # 如果没有指定include条件（白名单），默认整个数据集
    if include_conditions is not None:
        # 如果有include条件，只留下白名单的编号
        combines = zip(*[h5f[key][:] for key in condition_keys])
        included = np.asarray([e in include_conditions for e in combines])
    # 在白名单里除去黑名单
    if exclude_conditions is not None:
        combines = zip(*[h5f[key][:] for key in condition_keys])
        not_excluded = np.asarray([e not in exclude_conditions for e in combines])
        included = np.logical_and(included, not_excluded)
    idx = np.where(included)[0]
    if shuffle:
        idx = np.random.permutation(idx)
    if seg_ratio is not None:
        idx = np.random.permutation(idx)
        num = int(seg_ratio * length)
        idx = (idx[:num], idx[num:])
        if outfile is not None:
            np.savetxt('train_' + outfile, idx[0], fmt='%d', delimiter='\n',
                       header="condition key: %s, include: %s, exclude: %s" % (
                       condition_keys, include_conditions, exclude_conditions))
            np.savetxt('test_' + outfile, idx[1], fmt='%d', delimiter='\n',
                       header="condition key: %s, include: %s, exclude: %s" % (
                       condition_keys, include_conditions, exclude_conditions))
    else:
        if outfile is not None:
            np.savetxt(outfile, idx, fmt='%d', delimiter='\n',
                       header="condition key: %s, include: %s, exclude: %s" % (
                       condition_keys, include_conditions, exclude_conditions))
    return idx

def make_dataset_stft(directory, pattern='**/*.dat', outpath='LTE_dataset_5c_1202.h5'):
    '''
    生成全信号STFT预处理的h5
    :param dir:
    :param film:
    :return:
    '''
    h5f = h5py.File(outpath, 'w')

    docs = os.listdir(directory)
    features = []
    for idx, doc in enumerate(docs):

        filepath = directory + '\\' + doc

        files = sorted(glob(filepath + '/' + pattern, recursive=True))

        signal = []
        num_samples_a_class = 0
        n = 0
        for file in files:
            if n >= 10:
                break
            timer = time.time()
            sig = read_dat(file)

            samples = energy_detect_N_cut(sig)

            num_samples_a_file = len(samples)
            print(file, ' | num samples :', num_samples_a_file)
            num_samples_a_class += num_samples_a_file

            for i, sample in enumerate(samples):

                feat_mat = myfft1(sample, nperseg=64, nfft=256, noverlap=52)
                features.append(feat_mat)

                if i % 100 == 99 or i == num_samples_a_file - 1:
                    append_data_to_h5(h5f, np.stack(features), 'features')
                    append_data_to_h5(h5f, np.ones(len(features), dtype=np.int8) * idx, 'labels')

                    features.clear()
            n += 1
            print(file, ' | times :', time.time() - timer, ' s')

        print(idx, ' name:', doc, ' samples:', num_samples_a_class)

    h5f.close()

def make_dataset_stft2(directory, pattern='**/*.dat', outpath='LTE_dataset_5c_1202.h5'):
    '''
    生成全信号STFT预处理的h5
    :param dir:
    :param film:
    :return:
    '''
    h5f = h5py.File(outpath, 'w')

    docs = os.listdir(directory)
    features = []
    for idx, doc in enumerate(docs):

        filepath = directory + '/' + doc

        files = sorted(glob(filepath + '/' + pattern, recursive=True))

        signal = []
        num_samples_a_class = 0
        n = 0
        for file in files:
            if n >= 10:
                break
            timer = time.time()
            sig = read_dat(file)

            samples = energy_detect_N_cut(sig)

            num_samples_a_file = len(samples)
            print(file, ' | num samples :', num_samples_a_file)
            num_samples_a_class += num_samples_a_file

            for i, sample in enumerate(samples):

                feat_mat = myfft2(sample)
                features.append(feat_mat)

                if i % 100 == 99 or i == num_samples_a_file - 1:
                    append_data_to_h5(h5f, np.stack(features), 'features')
                    append_data_to_h5(h5f, np.ones(len(features), dtype=np.int8) * idx, 'labels')

                    features.clear()
            n += 1
            print(file, ' | times :', time.time() - timer, ' s')

        print(idx, ' name:', doc, ' samples:', num_samples_a_class)

    h5f.close()

def resize_dataset(inputpath='LTE_dataset_5c_1202.h5', outpath='LTE_dataset_5c_1202.h5'):
    '''
    生成全信号STFT预处理的h5
    :param dir:
    :param film:
    :return:
    '''
    input = h5py.File(inputpath, 'r')
    h5f = h5py.File(outpath, 'w')
    for key in input.keys():
        if key == 'labels':
            append_data_to_h5(h5f, input[key][:], 'labels')
        else:
            data_iter = input[key]
            batch_size = 100
            for s in range(0,len(data_iter),batch_size):
                e = s+batch_size
                #bottle_resized = resize(bottle, (140, 54))
                data = [resize(myfft1_norm(x),(64,64,4)) for x in data_iter[s:e]]

                append_data_to_h5(h5f, input[key][:], key)
    h5f.close()

def energy_detect_N_cut_return_dict(origin_signal, window_size=100, gate=1e3, num_win_per_sample=30):
    '''
    能量检测，时域切割
    :param origin_signal: numpy ndarray, shape = [Num, 2]，原始时域信号, I、Q两路
    :param window_size:
    :param gate:
    :return: numpy ndarray, shape = [n, window_size * num_win_per_sample, 2], 切割后的时域信号, I、Q两路
    '''

    analyze = origin_signal[:, 0] + np.asarray(1j, np.complex64) * origin_signal[:, 1]
    analyze = analyze - np.mean(analyze)

    energy_len = int(np.floor(len(analyze) / window_size))
    energy = np.zeros(energy_len)
    for i in range(energy_len):
        head = i * window_size
        tail = i * window_size + window_size
        energy[i] = np.linalg.norm(analyze[head:tail])

    indices = energy > gate
    samples = []
    pos = []
    mean_energy = []
    i = 0
    while i <= (len(energy) - num_win_per_sample):
        if ~indices[i]:
            i += 1
            continue
        flag = ~indices[i:i + num_win_per_sample]
        if np.any(flag):
            i += np.where(flag)[0][-1] + 1
        else:
            mean_energy.append(np.mean(energy[i:i + num_win_per_sample]))
            pos.append(i * window_size)
            samples.append(analyze[i * window_size:(i + num_win_per_sample) * window_size])
            i += num_win_per_sample

    return {'samples': samples, 'position': pos, 'energy': mean_energy}


def myfft2(signal, nperseg=64, nfft=128, noverlap=44,_resize = True):
    _, _, z = Sig.stft(signal, nperseg=nperseg, nfft=nfft, noverlap=noverlap, return_onesided=False)
    z = np.fft.fftshift(z, 0)[:,:nfft]
    z = z / np.max(abs(z))
    feature = np.zeros((z.shape[0], z.shape[1], 4), np.float32)
    feature[:, :, 0] = z.real/2+0.5
    feature[:, :, 1] = z.imag/2+0.5
    feature[:, :, 2] = np.log10(abs(z)+1e-8)
    feature[:, :, 2] = (feature[:, :, 2]-feature[:,:,2].min())/(-feature[:,:,2].min())
    feature[:, :, 3] = np.angle(z)/2/np.pi + 0.5
    if _resize:
        out = resize(feature,(64,64,4),anti_aliasing=False)
    else:
        out = feature
    return out

def myfft1(signal, nperseg=64, nfft=256, noverlap=52):
    _, _, z = Sig.stft(signal, nperseg=64, nfft=256, noverlap=52, return_onesided=False)
    z = np.fft.fftshift(z, 0)
    z = z / np.max(abs(z))
    feature = np.zeros((z.shape[0], z.shape[1], 4), np.float32)
    feature[:, :, 0] = z.real
    feature[:, :, 1] = z.imag
    feature[:, :, 2] = np.log10(abs(z))
    feature[:, :, 3] = np.angle(z)
    return feature

def stft_phase(signal, nperseg=64, nfft=256, noverlap=52):
    _, _, z = Sig.stft(signal, nperseg=64, nfft=256, noverlap=52, return_onesided=False)
    z = np.fft.fftshift(z, 0)
    phase = np.zeros((z.shape[0], z.shape[1]), np.float32)
    phase[:, :] = np.angle(z)
    return phase


def myfft_for_display_1(origin_signal,
                        nperseg=250,
                        nfft=512,
                        noverlap=200,
                        batch_size=1000000):
    analyze = origin_signal[:, 0] + np.asarray(1j, np.complex64) * origin_signal[:, 1]
    analyze = analyze - np.mean(analyze)
    l = len(origin_signal)
    features = []
    for s in range(0, l - 1, batch_size):
        e = min(s + batch_size, l)
        _, _, z = Sig.stft(analyze[s:e], nperseg=nperseg, nfft=nfft, noverlap=noverlap, return_onesided=False)
        z = np.fft.fftshift(z, 0)
        feature = 10 * np.log10(abs(z))
        features.append(feature)
    return features


def diplay_1(sig,
             pos,
             predict,
             enengy,
             figname='display',
             nperseg=250,
             step=50,
             nfft=512,
             batch_size=1000000):
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

    f = myfft_for_display_1(sig, nperseg=nperseg, nfft=nfft, noverlap=nperseg - step, batch_size=batch_size)
    for i in range(len(f)):

        samples_batch_indices = np.asarray(pos) < (i + 1) * batch_size & np.asarray(pos) >= i * batch_size

        p = np.asarray(pos)[samples_batch_indices] // step

        e = 10 * np.log10(np.asarray(enengy, np.float32)[samples_batch_indices])

        c = np.asarray(predict, np.int8)[samples_batch_indices]

        plt.figure(figsize=(180, 5))
        plt.imshow(f[i])
        for j in range(len(p)):
            color = ['red', 'black', 'yellow', 'white', 'aqua']
            plt.plot([p[j], p[j] + 30], [e[j], e[j]], color=color[c[j]], linewidth=10, label='predicted as ' + str(c))
        plt.legend()
        plt.savefig(figname + str(i) + '.jpg')


def append_data_to_h5(h5f, data, target):
    shape_list = list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0] = None  # 设置数组的第一个维度是0
        dataset = h5f.create_dataset(target, data=data, maxshape=tuple(shape_list), chunks=True)
        return
    else:
        dataset = h5f[target]
    len_old = dataset.shape[0]
    len_new = len_old + data.shape[0]
    shape_list[0] = len_new
    dataset.resize(tuple(shape_list))  # 修改数组的第一个维度
    dataset[len_old:len_new] = data  # 存入新的文件


def get_train_valid_indices(number_samples, train_percent=0.6, random_seed=666):
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

# DATA9_ROOT = '/media/ubuntu/Seagate Expansion Drive/data 9/电台数据'
DATA9_ROOT = '/media/ubuntu/9d99a77e-02ce-4e2b-a8a1-243cd4bdef7d/workplace/lwj/电台数据'
PATH_DICT = {'DATA9_ROOT':DATA9_ROOT}
for file in os.listdir(DATA9_ROOT):
    path = os.path.abspath(os.path.join(DATA9_ROOT,file))
    if os.path.isdir(path):
        PATH_DICT[file] = path


if __name__ == '__main__':
    # path = 'E:\\DATA\\Desktop\\LTE.m\\dat\\split'
    # make_dataset_stft(path)
    # make_dataset_fc('../../dataset/dat/s2')
    # make_dataset_fc(DATA9_ROOT)
    make_dataset_signal_fc(DATA9_ROOT, drop_abnormal_mean=True, outpath="dataset_signal_5000_new.h5")

