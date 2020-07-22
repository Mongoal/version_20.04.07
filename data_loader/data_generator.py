import numpy as np

from data_loader.h5data_reader import H5DataReader

# 减少io次数，一次读取BUFFER_SIZE个batch
BUFFER_SIZE = 32

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.h5_reader = H5DataReader(config.h5_data_path, mode='r', data_key=config.h5_data_key, label_key=config.h5_label_key, seed=config.h5_shuffle_seed)
        self.train_batch_generator = None
        self.test_batch_generator = None
        self.batch_generator = None



    def get_train_batch_generator(self, batch_size):
        '''

        获取遍历整个数据集的迭代器，每次取数据：
        generator = get_batch_generator(batch_size)
        for _ in it:
            data = next(generator)
            process(data)

        或use like : batch_x, batch_y = next(self.data.get_train_batch_generator(self.config.batch_size))
        Args:
            batch_size:

        Returns:

        '''
        if self.train_batch_generator is None:
            self.train_batch_generator = self.next_train_batch_generator(batch_size)
        return self.train_batch_generator

    def get_test_batch_generator(self, batch_size):
        '''
        获取遍历整个数据集的迭代器，每次取数据：
        generator = get_batch_generator(batch_size)
        for _ in it:
            data = next(generator)
            process(data)
        Args:
            batch_size:

        Returns:

        '''
        if self.test_batch_generator is None:
            self.test_batch_generator = self.next_test_batch_generator(batch_size)
        return self.test_batch_generator

    def get_batch_generator(self, batch_size):
        '''
        获取遍历整个数据集的迭代器，每次取数据：
        generator = get_batch_generator(batch_size)
        for _ in it:
            data = next(generator)
            process(data)
        Args:
            batch_size: 批大小

        Returns:迭代器
        '''
        if self.batch_generator is None:
            self.batch_generator = self.next_batch_generator(batch_size)
        return self.batch_generator


    def next_batch_generator(self, batch_size):
        '''
        return an generator
        Args:
            batch_size:

        Returns: batch_generator

        '''
        while True:
            buffer_x, buffer_y = self.h5_reader.get_shuffle_data(batch_size * BUFFER_SIZE)
            buffer_length = len(buffer_x)
            i = 0
            while(i < buffer_length):
                start = i
                i += batch_size
                yield buffer_x[start:i], buffer_y[start:i]


    def next_train_batch_generator(self, batch_size):
        '''
        return an generator
        Args:
            batch_size:

        Returns: batch_generator

        '''
        while True:
            buffer_x, buffer_y = self.h5_reader.get_train_batch(batch_size * BUFFER_SIZE)
            buffer_length = len(buffer_x)
            i = 0
            while(i < buffer_length):
                start = i
                i += batch_size
                yield buffer_x[start:i], buffer_y[start:i]

    def next_test_batch_generator(self, batch_size):
        '''
        return an generator
        Args:
            batch_size:

        Returns: batch_generator

        '''
        while True:
            buffer_x, buffer_y = self.h5_reader.get_test_batch(batch_size * BUFFER_SIZE)
            buffer_length = len(buffer_x)
            i = 0
            while(i < buffer_length):
                start = i
                i += batch_size
                yield buffer_x[start:i], buffer_y[start:i]


    def get_epoch_size(self, batch_size):
        return (self.h5_reader.length-1)//batch_size + 1
