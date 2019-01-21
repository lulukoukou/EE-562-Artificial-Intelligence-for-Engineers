import numpy as np
from config import *


class DataSet(object):
    def __init__(self, phase):
        assert phase == 'train' or phase == 'valid' or phase == 'test', 'invalid phase input'
        self.data = np.load(cifar10_dir+'%s_data.npy' % phase)
        self.label = np.load(cifar10_dir+'%s_label.npy' % phase)
        self.mean_image = np.load(cifar10_dir+'mean_image.npy')
        self.data_size = self.label.shape[0]
        self.index = np.arange(self.data_size)
        self.batch_size = BATCH_SIZE_TRAIN if phase != 'test' else BATCH_SIZE_TEST
        self.shuffle = phase == 'train'
        self.batch_per_epoch = self.data_size // self.batch_size
        self.cur_batch = 0
        self.cur_iteration = 0
        self.cur_epoch = 0

    def reset(self):
        self.cur_batch = 0
        self.cur_iteration = 0
        self.cur_epoch = 0

    def next_batch(self):
        if self.shuffle and self.cur_batch == 0:
            np.random.shuffle(self.index)

        batch_data = np.zeros((self.batch_size, 32, 32, 3), dtype=np.uint8)
        batch_label = np.zeros((self.batch_size, ), dtype=np.uint8)

        for i in range(self.batch_size):
            batch_data[i] = self.data[self.index[self.cur_batch * self.batch_size + i]]
            batch_label[i] = self.label[self.index[self.cur_batch * self.batch_size + i]]

        self.cur_batch += 1
        self.cur_iteration += 1

        if self.cur_batch == self.batch_per_epoch:
            self.cur_batch = 0
            self.cur_epoch += 1

        return batch_data, batch_label
