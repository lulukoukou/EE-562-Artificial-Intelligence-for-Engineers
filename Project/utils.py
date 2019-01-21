import numpy as np
from config import cifar10_dir


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_training_data():
    data = np.zeros((50000, 3, 32, 32), np.uint8)
    data[0    :10000, :] = unpickle(cifar10_dir + 'data_batch_1')[b'data'].reshape((-1, 3, 32, 32))
    data[10000:20000, :] = unpickle(cifar10_dir + 'data_batch_2')[b'data'].reshape((-1, 3, 32, 32))
    data[20000:30000, :] = unpickle(cifar10_dir + 'data_batch_3')[b'data'].reshape((-1, 3, 32, 32))
    data[30000:40000, :] = unpickle(cifar10_dir + 'data_batch_4')[b'data'].reshape((-1, 3, 32, 32))
    data[40000:50000, :] = unpickle(cifar10_dir + 'data_batch_5')[b'data'].reshape((-1, 3, 32, 32))
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 1, 2)

    mean_image = np.mean(data, axis=0)

    labels = np.zeros((50000,), np.uint8)
    labels[0    :10000] = np.array(unpickle(cifar10_dir + 'data_batch_1')[b'labels']).astype(np.uint8)
    labels[10000:20000] = np.array(unpickle(cifar10_dir + 'data_batch_2')[b'labels']).astype(np.uint8)
    labels[20000:30000] = np.array(unpickle(cifar10_dir + 'data_batch_3')[b'labels']).astype(np.uint8)
    labels[30000:40000] = np.array(unpickle(cifar10_dir + 'data_batch_4')[b'labels']).astype(np.uint8)
    labels[40000:50000] = np.array(unpickle(cifar10_dir + 'data_batch_5')[b'labels']).astype(np.uint8)

    np.save(cifar10_dir+'training_data.npy', data)
    np.save(cifar10_dir+'training_label.npy', labels)
    np.save(cifar10_dir+'mean_image.npy', mean_image)
    print("training data saved to npy format")


def load_test_data():
    data = unpickle(cifar10_dir + 'test_batch')[b'data'].reshape((-1, 3, 32, 32))
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 1, 2)
    labels = np.array(unpickle(cifar10_dir + 'test_batch')[b'labels']).astype(np.uint8)
    np.save(cifar10_dir+'test_data.npy', data)
    np.save(cifar10_dir+'test_label.npy', labels)
    print("test data saved to npy format")


def load_label_names():
    return unpickle(cifar10_dir + 'batches.meta')[b'label_names']  # list


def train_valid_split():
    data = np.load(cifar10_dir+'training_data.npy')
    train = data[1:45000]
    valid = data[45000:]
    np.save(cifar10_dir+'train_data.npy', train)
    np.save(cifar10_dir+'valid_data.npy', valid)

    label = np.load(cifar10_dir + 'training_label.npy')
    label_train = label[1:45000]
    label_valid = label[45000:]
    np.save(cifar10_dir+'train_label.npy', label_train)
    np.save(cifar10_dir+'valid_label.npy', label_valid)

    print("split done")


load_training_data()
load_test_data()
train_valid_split()