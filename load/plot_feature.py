from matplotlib import pyplot as plt
from data_loader.h5data_reader import H5DataReader

import os
def plot(features,i):
    for feature in features:
        plt.plot(feature[:,i])
        plt.show()
        plt.clf()
def savefig(features,i,name):
    k = 0
    plt.figure(figsize=(20,6))
    for feature in features:
        k+=1
        plt.plot(feature[:1000,i],linewidth=1)
        plt.savefig(name+str(k)+'.png')
        plt.clf()
def plot2d(features,i):
    for feature in features:
        plt.imshow(feature[:,:,i])
        plt.show()
        plt.clf()

if __name__ == '__main__':
    stft_path = '../../dataset/LTE_dataset_stft_256x256x4_3c_1216.h5'
    signal_path = '../../dataset/LTE_origin_3240_dataset_5c_10s_1202.h5'
    print(os.path.abspath(signal_path))
    # stft_reader = H5DataReader(stft_path,)
    # sig_reader = H5DataReader(signal_path, 'r', 'signals')
    sig_reader = H5DataReader(stft_path, 'r')
    batch,_ = sig_reader.get_shuffle_data(1)
    print(batch[0].shape)
    plot2d(batch,2)
