import tensorflow as tf
import numpy as np

from config import *
from datalayer import DataSet
from network import ConvNet


def train():
    lr = LR
    data = DataSet('train')
    valid_data = DataSet('valid')
    net = ConvNet(net_name, BATCH_SIZE_TRAIN)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)

        while data.cur_iteration < MAX_ITERATION:
            if data.cur_iteration == 32000 or data.cur_iteration == 48000:
                lr /= 10
            batch_data, batch_label = data.next_batch()

            feed_dict = {net.ph_inputs: batch_data,
                         net.ph_labels: batch_label,
                         net.ph_mean: data.mean_image,
                         net.ph_lr: lr,
                         net.ph_training: True}

            _, _loss, _summary = sess.run([net.optimizer, net.loss, net.summary], feed_dict=feed_dict)
            if data.cur_iteration % 100 == 0:
                print("ite:%5d  ep:%4d  loss:%4.4f" % (data.cur_iteration, data.cur_epoch, _loss))
                summary_writer.add_summary(_summary, data.cur_iteration)
            if data.cur_iteration % 5000 == 0:
                ckpt_fname = 'train_model.ckpt'
                ckpt_full_fname = os.path.join(model_dir, ckpt_fname)
                saver.save(sess, ckpt_full_fname, data.cur_epoch)

                n_correct = 0
                while valid_data.cur_epoch == 0:
                    batch_data, batch_label = valid_data.next_batch()
                    feed_dict = {net.ph_inputs: batch_data,
                                 net.ph_labels: batch_label,
                                 net.ph_mean: valid_data.mean_image,
                                 net.ph_training: False}
                    _pred, _loss = sess.run([net.pred, net.loss], feed_dict=feed_dict)
                    _pred = _pred.astype(np.uint8)
                    correct = np.zeros(data.batch_size, dtype=np.uint8)
                    correct[_pred == batch_label] = 1
                    n_correct += sum(correct)
                pctg = n_correct / (valid_data.batch_per_epoch*valid_data.batch_size)
                _sum = sess.run(net.summary_valid, feed_dict={net.ph_pctg: pctg})
                summary_writer.add_summary(_sum, data.cur_iteration)
                valid_data.reset()
                print('validation: %.4f' % pctg)


def test():
    data = DataSet('test')
    net = ConvNet(net_name, BATCH_SIZE_TEST)
    n_correct = 0

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("load model -- succeed")
        else:
            print("load model -- fail")
        while data.cur_epoch == 0:
            batch_data, batch_label = data.next_batch()
            feed_dict = {net.ph_inputs: batch_data,
                         net.ph_labels: batch_label,
                         net.ph_mean: data.mean_image,
                         net.ph_training: False}
            _pred, _loss = sess.run([net.pred, net.loss], feed_dict=feed_dict)
            _pred = _pred.astype(np.uint8)
            correct = np.zeros(data.batch_size, dtype=np.uint8)
            correct[_pred == batch_label] = 1
            n_correct += sum(correct)

        print('test: %.4f' % (n_correct / (data.batch_size * data.batch_per_epoch)))


if __name__ == '__main__':
    import time

    t = time.time()
    train()
    print("=========================================================================")
    print('training time: ' + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
    print("=========================================================================")

    tf.reset_default_graph()
    test()
