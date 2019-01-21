import os
import platform

net_name = 'vgg9'
# net_name = 'vgg11'
# net_name = 'vgg15'
# net_name = 'plain32'
# net_name = 'resnet32'
# net_name = 'resnet110'

LR = 0.001
n_gpu = 7

os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%n_gpu

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
MAX_ITERATION = 64000

cifar10_dir = ('D:' if platform.system() == 'Windows' else '') + '/data/chang/cifar-10/data/'
result_dir = ('D:' if platform.system() == 'Windows' else '') + '/data/chang/cifar-10/results/'\
             + net_name+'/lr%.4f'%LR + '_gpu%d'%n_gpu + '/'
tb_dir = result_dir+'tb'
model_dir = result_dir + 'model'

if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)