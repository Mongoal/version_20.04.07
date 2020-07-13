# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:47:33 2019
author: Liwei Huang dr_huanglw@163.com
"""
import os
import numpy as np
from glob import glob
import random

def read_data(directory, pattern='**/*.txt'):

    docs = os.listdir(directory)
    signals = []
    i = 0
    for doc in docs:
        filepath = directory + '\\' + doc
        files = sorted(glob(filepath + '/' + pattern, recursive=True))
        signal = []
        for file in files:
            file = open(file, 'r')
            list_of_all_lines = file.readlines()
            lines = np.array(list_of_all_lines, dtype=np.float32)
            if lines.shape[0]< 13000 and lines.shape[0]> 8000:
                signal.append(lines)
            file.close()
        if len(signal) > 300:
            signals.append(signal)
            # test
            print(i, ' name:', doc, ' files:', len(signal))
            i = i + 1
    labels=[]
    feature=[]
    n_labels=len(signals)
    for idx,signal in enumerate(signals): #[ Class1_list([sig1],[sig2],[sigM]),Class2_list(),...,ClassN_list()]
        for sig in signal: #signal = Classn_list([sig1],[sig2],[sigM])
            feat_mat=get_features(sig, window_length=500, window_step=56, NFFT=446, max_frames=224) #input: sig1 ,shape = 13040 . return : array shape = 256*256
            feature.append(feat_mat)
            labels.append(idx)
    return feature,labels,n_labels

def get_features(sig , window_length, window_step, NFFT, max_frames=128):
    '''
    主要从输入采样序列中读入数据，然后转换为时频域特征
    sig：输入序列；rate：采样频率；window_length：每一帧的时间窗口长度；window_step：每一帧的时间重叠步长
    '''
    feat_mat = []
    for i in range(max_frames):
        start = window_step * i
        end = start + window_length
        slice_sig = sig[start:end]
        feature = STFT(slice_sig, NFFT)
        feat_mat.append(feature)
    feat_mat = np.array(feat_mat, dtype=float)
    return feat_mat

def STFT(frames, NFFT):
    '''计算每一帧经过FFT变换以后的频谱的幅度，frames的大小为N*L,则返回矩阵的大小为N*(NFFT//2+1)
    参数说明：
    frames:帧矩阵
    NFFT:FFT变换的数组大小,如果帧长度小于NFFT，则帧的其余部分用0填充铺满
    '''
    complex_spectrum=np.fft.rfft(frames,NFFT) #对frames进行FFT变换
    complex_spectrum=np.absolute(complex_spectrum)  #返回频谱的幅度值
    return 1.0/NFFT * np.square(complex_spectrum) #功率谱等于每一点的幅度平方/NFFT  

def pre_emphasis(signal,coefficient=0.95):
    '''对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return np.append(signal[0],signal[1:]-coefficient*signal[:-1])

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]

def generate_batches(labels, batch_size, balance=True):
    
    indices = dict()
    for idx, label in enumerate(labels):
        if label not in indices:
            indices[label] = []
        indices[label].append(idx)
    count = dict()
    ap=[]
    if balance:
        len_ap_last = len(ap)
        maxlength = 1500
        for key in indices.keys():
            len_ap_last = len(ap)
            if len(indices[key])>maxlength:
                randomi = np.arange(len(indices[key]))
                np.random.shuffle((randomi))
                range_i = randomi[:maxlength]
            else:
                range_i = range(len(indices[key]) -1)
            for i in range_i:
                if len(indices[key])-i > maxlength:
                    randomj = np.array(range(i+1, len(indices[key])))
                    np.random.shuffle((randomj))
                    range_j = randomj[:maxlength]
                else:
                    range_j = range(i+1,len(indices[key]))
                for j in range_j:
                    rpts = maxlength/len(indices[key])
                    rpts = int(rpts*rpts/2)
                    if rpts > 4:
                        for _ in range(rpts):
                            ap.append([key, [indices[key][i], indices[key][j]]])
                    else:
                        ap.append([key, [indices[key][i], indices[key][j]]])
            print(key,' num_ap :', len(ap)-len_ap_last)
    else:
        len_ap_last = len(ap)
        for key in indices.keys():
            for i in range(len(indices[key])-1):
                for j in range(i+1,len(indices[key])):
                    ap.append([key, [indices[key][i],indices[key][j]]])
        print(key, ' num_ap :', len(ap) - len_ap_last)
    
    #随机选择1/3个ap
    ap_num = int(batch_size/3)
    batch_num = int(len(ap)/ap_num)
    random.shuffle(ap)
    batches=[]
    for i in range(batch_num):
        list_aps = ap[i*ap_num:(i+1)*ap_num]
        n_labels = list(set(indices.keys())-set([x[0] for x in list_aps]))  #不在list中的label
        while len(n_labels) == 0:# Linwanjie 9/16 add, 处理negative为空的异常
            p = np.random.randint(batch_num)
            list_aps = ap[p * ap_num:(p + 1) * ap_num]
            n_labels = list(set(indices.keys()) - set([x[0] for x in list_aps]))  # 不在list中的label
            # print(i, p)
        batch_ap = set(sum([x[1] for x in list_aps],[])) #已选的anchor和Positive
        num_negative = batch_size - len(batch_ap)
        houxuan = sum([indices[key] for key in n_labels],[]) #negative的候选集合
        batch_negative = np.random.choice(list(set(houxuan)), num_negative)
        triplet = list(batch_ap)+list(batch_negative)
        # print(i,n_labels,batch_ap,batch_negative,len(triplet)) # test
        batches.append(triplet)
        # count
        for key in n_labels:
            if key not in count:
                count[key] = 0
            count[key] =count[key]+1
    print(count)
    if batch_num * ap_num < len(ap):
        list_aps = ap[batch_num * ap_num:]
        n_labels = list(set(indices.keys())-set([x[0] for x in list_aps]))  #不在list中的label
        batch_ap = set(sum([x[1] for x in list_aps],[])) #已选的anchor和Positive
        num_negative = batch_size - len(batch_ap)
        houxuan = sum([indices[key] for key in n_labels],[]) #negative的候选集合
        batch_negative = np.random.choice(list(set(houxuan)),num_negative)
        triplet = list(batch_ap)+list(batch_negative)
        batches.append(triplet)
    return batches