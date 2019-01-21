import tensorflow as tf
from config import *
import numpy as np


class ConvNet(object):
    def __init__(self, name, batch_size):
        self.batch_size = batch_size
        self.ph_inputs = tf.placeholder(tf.uint8, shape=(self.batch_size, 32, 32, 3), name='ph_input')
        self.ph_labels = tf.placeholder(tf.uint8, shape=(self.batch_size,), name='ph_labels')
        self.ph_mean = tf.placeholder(tf.float64, shape=(32, 32, 3), name='ph_mean')
        self.ph_lr = tf.placeholder(tf.float64, shape=None, name='ph_lr')
        self.ph_training = tf.placeholder(tf.bool, shape=None, name='ph_training')
        self.ph_pctg = tf.placeholder(tf.float64, shape=None, name='ph_pctg')

        name_net = {'resnet32': self.resnet32,
                    'resnet110': self.resnet110,
                    'plain32': self.plain32,
                    'vgg11': self.vgg11,
                    'vgg15': self.vgg15,
                    'vgg9': self.vgg9}
        self.net = name_net[name]

        self.logits = self.net(self.ph_inputs, self.ph_mean)
        self.prob = tf.nn.softmax(self.logits)
        self.pred = tf.argmax(self.prob, axis=1)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(self.ph_labels, tf.int32), logits=self.logits, name='ce_loss'))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.ph_lr).minimize(self.loss)
        self.summary = tf.summary.scalar("loss", self.loss)
        self.summary_valid = tf.summary.scalar("pctg", self.ph_pctg)

    def vgg9(self, inputs, mean_image):
        def _block(num_layers, num_filters, inputs):
            conv = inputs
            for i in range(num_layers):
                conv = tf.layers.conv2d(inputs, num_filters, (3, 3), 1, 'same')
                conv = tf.layers.batch_normalization(conv, training=self.ph_training)
                conv = tf.nn.relu(conv)
            return conv

        inputs = tf.cast(inputs, tf.float32)
        mean_image = tf.cast(mean_image, tf.float32)
        inputs -= mean_image

        with tf.variable_scope('vgg9', reuse=tf.AUTO_REUSE):
            conv = _block(2, 16, inputs)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(2, 32, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(2, 64, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)

            # FINAL FEATURE MAP IS 4X LARGER THAN VGG11, VGG15
            length = int(np.prod(conv.get_shape()[1:]))
            fc = tf.reshape(conv, [-1, length])
            fc = tf.layers.dense(fc, 64)
            fc = tf.layers.batch_normalization(fc, training=self.ph_training)
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(fc, 64)
            fc = tf.layers.batch_normalization(fc, training=self.ph_training)
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(fc, 10)
        return fc

    def vgg11(self, inputs, mean_image):
        def _block(num_layers, num_filters, inputs):
            conv = inputs
            for i in range(num_layers):
                conv = tf.layers.conv2d(inputs, num_filters, (3, 3), 1, 'same')
                conv = tf.layers.batch_normalization(conv, training=self.ph_training)
                conv = tf.nn.relu(conv)
            return conv

        inputs = tf.cast(inputs, tf.float32)
        mean_image = tf.cast(mean_image, tf.float32)
        inputs -= mean_image

        with tf.variable_scope('vgg11', reuse=tf.AUTO_REUSE):
            conv = _block(2, 16, inputs)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(2, 32, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(2, 64, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(2, 64, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)

            length = int(np.prod(conv.get_shape()[1:]))
            fc = tf.reshape(conv, [-1, length])
            fc = tf.layers.dense(fc, 64)
            fc = tf.layers.batch_normalization(fc, training=self.ph_training)
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(fc, 64)
            fc = tf.layers.batch_normalization(fc, training=self.ph_training)
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(fc, 10)
        return fc

    def vgg15(self, inputs, mean_image):
        def _block(num_layers, num_filters, inputs):
            conv = inputs
            for i in range(num_layers):
                conv = tf.layers.conv2d(inputs, num_filters, (3, 3), 1, 'same')
                conv = tf.layers.batch_normalization(conv, training=self.ph_training)
                conv = tf.nn.relu(conv)
            return conv

        inputs = tf.cast(inputs, tf.float32)
        mean_image = tf.cast(mean_image, tf.float32)
        inputs -= mean_image

        with tf.variable_scope('vgg11', reuse=tf.AUTO_REUSE):
            conv = _block(3, 16, inputs)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(3, 32, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(3, 64, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)
            conv = _block(3, 64, conv)
            conv = tf.layers.max_pooling2d(conv, (2, 2), 2)

            length = int(np.prod(conv.get_shape()[1:]))
            fc = tf.reshape(conv, [-1, length])
            fc = tf.layers.dense(fc, 64)
            fc = tf.layers.batch_normalization(fc, training=self.ph_training)
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(fc, 64)
            fc = tf.layers.batch_normalization(fc, training=self.ph_training)
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(fc, 10)
        return fc

    def plain32(self, inputs, mean_image):
        def residual_block_2layers(inputs, num_filters, sz_filter, downsample=False, name=None):
            assert len(num_filters) == 2
            N, H, W, C = inputs.get_shape().as_list()
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                residue = tf.layers.conv2d(inputs, num_filters[0], (sz_filter, sz_filter),
                                           1 if not downsample else 2, 'same', name='conv1')
                residue = tf.layers.batch_normalization(residue, training=self.ph_training)
                residue = tf.nn.relu(residue)
                residue = tf.layers.conv2d(residue, num_filters[1], (sz_filter, sz_filter),
                                           1, 'same', name='conv2')

                outputs = tf.layers.batch_normalization(residue, training=self.ph_training)
                outputs = tf.nn.relu(outputs)

                return outputs

        inputs = tf.cast(inputs, tf.float32)
        mean_image = tf.cast(mean_image, tf.float32)
        inputs -= mean_image

        with tf.variable_scope('plain32', reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(inputs, 16, (3, 3), 1, 'same', name='conv0')

            for i in range(5):
                conv = residual_block_2layers(conv, num_filters=[16, 16], sz_filter=3, downsample=False,
                                              name='res_blk_0_%d'%i)

            conv = residual_block_2layers(conv, num_filters=[32, 32], sz_filter=3, downsample=True,
                                            name='res_blk_1_0')
            for i in range(1, 5):
                conv = residual_block_2layers(conv, num_filters=[32, 32], sz_filter=3, downsample=False,
                                              name='res_blk_1_%d' % i)

            conv = residual_block_2layers(conv, num_filters=[64, 64], sz_filter=3, downsample=True,
                                          name='res_blk_2_0')
            for i in range(1, 5):
                conv = residual_block_2layers(conv, num_filters=[64, 64], sz_filter=3, downsample=False,
                                              name='res_blk_2_%d' % i)

            _, h, w, _ = conv.get_shape().as_list()
            gap = tf.layers.average_pooling2d(conv, (h, w), 1, 'valid', name='gap')
            logits = tf.layers.dense(tf.squeeze(gap), 10, name='fc')
            assert len(logits.get_shape().as_list()) == 2

            return logits

    def resnet32(self, inputs, mean_image):
        def residual_block_2layers(inputs, num_filters, sz_filter, downsample=False, name=None):
            assert len(num_filters) == 2
            N, H, W, C = inputs.get_shape().as_list()
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                residue = tf.layers.conv2d(inputs, num_filters[0], (sz_filter, sz_filter),
                                           1 if not downsample else 2, 'same', name='conv1')
                residue = tf.layers.batch_normalization(residue, training=self.ph_training)
                residue = tf.nn.relu(residue)
                residue = tf.layers.conv2d(residue, num_filters[1], (sz_filter, sz_filter),
                                           1, 'same', name='conv2')

                identity = tf.layers.max_pooling2d(inputs, (2, 2), 2, 'SAME') \
                    if downsample else inputs
                if C != num_filters[1]:
                    pad_chl = (num_filters[1] - C) // 2
                    identity = tf.pad(identity, ([0, 0], [0, 0], [0, 0], [pad_chl, pad_chl]))

                outputs = identity + residue
                outputs = tf.layers.batch_normalization(outputs, training=self.ph_training)
                outputs = tf.nn.relu(outputs)

                return outputs

        inputs = tf.cast(inputs, tf.float32)
        mean_image = tf.cast(mean_image, tf.float32)
        inputs -= mean_image

        with tf.variable_scope("resnet32", reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(inputs, 16, (3, 3), 1, 'same', name='conv0')

            for i in range(5):
                conv = residual_block_2layers(conv, num_filters=[16, 16], sz_filter=3, downsample=False,
                                              name='res_blk_0_%d'%i)

            conv = residual_block_2layers(conv, num_filters=[32, 32], sz_filter=3, downsample=True,
                                            name='res_blk_1_0')
            for i in range(1, 5):
                conv = residual_block_2layers(conv, num_filters=[32, 32], sz_filter=3, downsample=False,
                                              name='res_blk_1_%d' % i)

            conv = residual_block_2layers(conv, num_filters=[64, 64], sz_filter=3, downsample=True,
                                          name='res_blk_2_0')
            for i in range(1, 5):
                conv = residual_block_2layers(conv, num_filters=[64, 64], sz_filter=3, downsample=False,
                                              name='res_blk_2_%d' % i)

            _, h, w, _ = conv.get_shape().as_list()
            gap = tf.layers.average_pooling2d(conv, (h, w), 1, 'valid', name='gap')
            logits = tf.layers.dense(tf.squeeze(gap), 10, name='fc')
            assert len(logits.get_shape().as_list()) == 2

            return logits

    def resnet110(self, inputs, mean_image):
        def residual_block_3layers(inputs, num_filters, sz_filter, downsample=False, name=None):
            assert len(num_filters) == 3
            N, H, W, C = inputs.get_shape().as_list()
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                residue = tf.layers.conv2d(inputs, num_filters[0], (1, 1),
                                           1, 'same', name='conv1')
                residue = tf.layers.batch_normalization(residue, training=self.ph_training)
                residue = tf.nn.relu(residue)
                residue = tf.layers.conv2d(residue, num_filters[1], (sz_filter, sz_filter),
                                           1 if not downsample else 2, 'same', name='conv2')
                residue = tf.layers.batch_normalization(residue, training=self.ph_training)
                residue = tf.nn.relu(residue)
                residue = tf.layers.conv2d(residue, num_filters[2], (1, 1), 1, 'same', name='conv3')

                identity = tf.layers.max_pooling2d(inputs, (2, 2), 2, 'SAME') \
                    if downsample else inputs
                if C != num_filters[2]:
                    pad_chl = (num_filters[2] - C) // 2
                    identity = tf.pad(identity, ([0, 0], [0, 0], [0, 0], [pad_chl, pad_chl]))

                outputs = identity + residue
                outputs = tf.layers.batch_normalization(outputs, training=self.ph_training)
                outputs = tf.nn.relu(outputs)
                return outputs

        inputs = tf.cast(inputs, tf.float32)
        mean_image = tf.cast(mean_image, tf.float32)
        inputs -= mean_image

        with tf.variable_scope('resnet110', reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(inputs, 16, (3, 3), 1, 'same', name='conv0')

            for i in range(12):
                conv = residual_block_3layers(conv, num_filters=[16, 16, 64], sz_filter=3, downsample=False,
                                              name='res_blk_0_%d'%i)

            conv = residual_block_3layers(conv, num_filters=[32, 32, 128], sz_filter=3, downsample=True,
                                            name='res_blk_1_0')
            for i in range(1, 12):
                conv = residual_block_3layers(conv, num_filters=[32, 32, 128], sz_filter=3, downsample=False,
                                              name='res_blk_1_%d' % i)

            conv = residual_block_3layers(conv, num_filters=[64, 64, 256], sz_filter=3, downsample=True,
                                          name='res_blk_2_0')
            for i in range(1, 12):
                conv = residual_block_3layers(conv, num_filters=[64, 64, 256], sz_filter=3, downsample=False,
                                              name='res_blk_2_%d' % i)

            _, h, w, _ = conv.get_shape().as_list()
            gap = tf.layers.average_pooling2d(conv, (h, w), 1, 'valid', name='gap')
            logits = tf.layers.dense(tf.squeeze(gap), 10, name='fc')
            assert len(logits.get_shape().as_list()) == 2

            return logits
