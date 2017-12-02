import time
import sys
import os

import cifar_input
import numpy as np
import resnet_model_basic as resnet_model
import tensorflow as tf
import data.cifar10_data as cifar10_data
import data2.cifar10_data as cifar_10data2

import json

from worker_I2L import worker_I2L, lr_I2L
from worker_L2I import worker_L2I
import argparse
import time

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=2, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('--valid_interval', type=int, default=1, help='Every how many epochs to valid?')
parser.add_argument('-r', '--load_params', type=str, default=None, help='The detailed model name')


# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('--trade_off_I2L', type=float, default=5e-3, help='the consistence tradeoff')
parser.add_argument('--trade_off_L2I', type=float, default=0.3, help='the consistence tradeoff')
parser.add_argument('-w', '--use_wide_resnet', dest='use_wide_resnet', action='store_true', help='Condition generative model on labels?')
parser.add_argument('--show_interval', type=int, default=100, help='Batch size during training per GPU')
parser.add_argument('--steal_params_L2I', type=str, default=None, help='Provide the file, which stores the warm values of L2I')
parser.add_argument('--steal_params_I2L', type=str, default=None, help='Provide the file, which stores the warm values of I2L')
parser.add_argument('--oneside', dest='oneside', type=str, default=None, help='None | I2L | L2I')

# optimization
parser.add_argument('--learning_rate_I2L', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--L2I_normalization', dest='L2I_normalization', action='store_true', help='Use L2I normalization')
parser.add_argument('--L2IuseSGD', dest='L2IuseSGD', action='store_true', help='Whether to use pure SGD to tune L2I')
parser.add_argument('--useSoftLabel', type=int, default=0, help='0: no use | 1: use | 2: -0.1')
# Activate "useSoftLabel" or not does not make significant differences. So my suggestion is that we do not need it. Also, I did not test useSoftLabel under multiple GPU settings
parser.add_argument('--bias', type=float, default=0.0, help='introduce the bias')


# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('--mode', type=str, default='train', help='train | I2L | L2I | ImgGen' )

# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

DataLoader = cifar10_data.DataLoader
DataLoader_train = cifar_10data2.DataLoader
rng = np.random.RandomState(args.seed)
train_data_iterator = DataLoader_train(args.data_dir, 'train', args.batch_size * args.nr_gpu,
                                       './cifar10_data/cifar10-LMscore',
                                       rng=rng, shuffle=True, return_labels=True)
test_data_iterator  = DataLoader(args.data_dir, 'test',  args.batch_size * args.nr_gpu, shuffle=False, return_labels=True)


class monitor(object):
  def __init__(self):
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    self.Worker_L2I = worker_L2I(args, train_data_iterator.get_num_labels(), train_data_iterator.get_observation_size())
    self.Worker_I2L = worker_I2L(args)

    self.image_LM = [tf.placeholder(tf.float32, shape=(args.batch_size,)) for _ in range(args.nr_gpu)]
    self.trade_off_I2L = tf.placeholder(tf.float32, shape=())
    self.trade_off_L2I = tf.placeholder(tf.float32, shape=())

    self.I2L_grads = []
    self.train_uidx = 0
    self._build_onestep(oneside=args.oneside)

    self.lr_l2i = self.Worker_L2I.args.learning_rate
    self.current_epoch = 0

    self.assign_op = lambda ref_, val_: tf.assign(ref_, val_)

  def get_I2L_lr(self):
    if args.use_wide_resnet:
      step_wise = [60, 120, 160]
      #step_wise = [51000, 76000, 102000] # counted by iter with batch_size 100
      if self.current_epoch < step_wise[0]:
        return args.learning_rate_I2L
      elif self.current_epoch < step_wise[1]:
        return args.learning_rate_I2L * 0.2
      elif self.current_epoch < step_wise[2]:
        return args.learning_rate_I2L * 0.04
      else:
        return args.learning_rate_I2L * 0.008
    else:
      step_wise = [102, 153, 204]
      #step_wise = [51000, 76000, 102000] # counted by iter with batch_size 100
      if self.current_epoch < step_wise[0]:
        return args.learning_rate_I2L
      elif self.current_epoch < step_wise[1]:
        return args.learning_rate_I2L * 0.1
      elif self.current_epoch < step_wise[2]:
        return args.learning_rate_I2L * 0.01
      else:
        return args.learning_rate_I2L * 0.001

  def get_L2I_lr(self):
    self.lr_l2i *= self.Worker_L2I.args.lr_decay
    return self.lr_l2i

  def __del__(self):
    self.sess.close()

  def _build_onestep(self, oneside=None):
    # Calculate all the costs and gradients
    # Let us NOT use weight decay, since we have aleardy had a regularization term
    # self.weightDecay_I2L = self.Worker_I2L.model.GetWeightDecay()
    self.nlls_I2L = self.Worker_I2L.model.nlls
    self.soft_labels = self.Worker_I2L.model.predictions # this is the soft labels [optional, may not use it]
    nlls_L2I, loss_gen_test = self.Worker_L2I.GetLoss()

    nlls_L2I_train_bpd_list, nlls_L2I_test_bpd_list, consistent_loss_list = \
      [None for _ in xrange(args.nr_gpu)], [None for _ in xrange(args.nr_gpu)], [None for _ in xrange(args.nr_gpu)]
    overall_cost_I2L_list, overall_cost_L2I_list, nlls_I2L_batchMean_list = \
      [None for _ in xrange(args.nr_gpu)], [None for _ in xrange(args.nr_gpu)], [None for _ in xrange(args.nr_gpu)]
    grads_I2L_list, grads_L2I_list = [None for _ in xrange(args.nr_gpu)], [None for _ in xrange(args.nr_gpu)]

    for i in range(args.nr_gpu):
      with tf.device('/gpu:%d' % i):
        nlls_L2I_train_bpd = tf.reduce_mean(nlls_L2I[i]) / (np.log(2.) * 32 * 32 * 3 )
        nlls_L2I_test_bpd = tf.reduce_mean(loss_gen_test[i]) / (np.log(2.) * 32 * 32 * 3 * args.batch_size)
        if args.L2I_normalization:
          consistent_loss = tf.reduce_mean(
            (self.image_LM[i] * np.log(2.) + self.nlls_I2L[i] + tf.log(0.1) - nlls_L2I[i] / (32. * 32 * 3)) ** 2.)
        else:
          consistent_loss = tf.reduce_mean(
            (self.image_LM[i] * np.log(2.) + (self.nlls_I2L[i] + tf.log(0.1) - nlls_L2I[i]) / 3072. + args.bias) ** 2.)

        nlls_L2I_train_bpd_list[i] = nlls_L2I_train_bpd
        nlls_L2I_test_bpd_list[i] = nlls_L2I_test_bpd
        consistent_loss_list[i] = consistent_loss
        nlls_I2L_batchMean = tf.reduce_mean(self.nlls_I2L[i])
        overall_cost_I2L = nlls_I2L_batchMean + (self.trade_off_I2L ** 2.) * consistent_loss
        overall_cost_L2I = nlls_L2I_train_bpd + (self.trade_off_L2I ** 2.) * consistent_loss
        nlls_I2L_batchMean_list[i] = nlls_I2L_batchMean
        overall_cost_I2L_list[i] = overall_cost_I2L
        overall_cost_L2I_list[i] = overall_cost_L2I

        if oneside is None:
          grads_I2L_list[i] = tf.gradients(overall_cost_I2L, self.Worker_I2L.model.trainable_variables)
          grads_L2I_list[i] = tf.gradients(overall_cost_L2I, self.Worker_L2I.all_params)
        elif oneside == 'I2L':
          grads_I2L_list[i] = tf.gradients(overall_cost_I2L, self.Worker_I2L.model.trainable_variables)
        elif oneside == 'L2I':
          grads_L2I_list[i] = tf.gradients(overall_cost_L2I, self.Worker_L2I.all_params)

    with tf.device('/gpu:0'):
      for i in range(1, args.nr_gpu):
        nlls_L2I_train_bpd_list[0] += nlls_L2I_train_bpd_list[i]
        nlls_L2I_test_bpd_list[0] += nlls_L2I_test_bpd_list[i]
        consistent_loss_list[0] += consistent_loss_list[i]
        overall_cost_I2L_list[0] += overall_cost_I2L_list[i]
        overall_cost_L2I_list[0] += overall_cost_L2I_list[i]
        nlls_I2L_batchMean_list[0] += nlls_I2L_batchMean_list[i]

        if oneside != 'L2I':
          for j in range(len(grads_I2L_list[0])):
            grads_I2L_list[0][j] += grads_I2L_list[i][j]
        if oneside != 'I2L':
          for j in range(len(grads_L2I_list[0])):
            grads_L2I_list[0][j] += grads_L2I_list[i][j]

      if oneside != 'L2I':
        for j in range(len(grads_I2L_list[0])):
          grads_I2L_list[0][j] /= (args.nr_gpu * 1.)

      if oneside != 'I2L':
        for j in range(len(grads_L2I_list[0])):
          grads_L2I_list[0][j] /= (args.nr_gpu * 1.)


      if oneside is None:
        self.Worker_I2L.model.Update(grads_I2L_list[0])
        self.Worker_L2I.Update(grads_L2I_list[0])
      elif oneside == 'I2L':
        self.Worker_I2L.model.Update(grads_I2L_list[0])
      elif oneside == 'L2I':
        self.Worker_L2I.Update(grads_L2I_list[0])

    self.nlls_L2I_train_bpd = nlls_L2I_train_bpd_list[0] / args.nr_gpu
    self.nlls_L2I_test_bpd = nlls_L2I_test_bpd_list[0] / args.nr_gpu
    self.consistent_loss = consistent_loss_list[0] /args.nr_gpu
    self.nlls_I2L_batchMean = nlls_I2L_batchMean_list[0] / args.nr_gpu
    self.overall_cost_I2L = overall_cost_I2L_list[0] / args.nr_gpu
    self.overall_cost_L2I = overall_cost_L2I_list[0] / args.nr_gpu

    # Build the sampler
    self.Worker_L2I.build_sample_from_model()

  def step(self, images, labels, LMscores, currEpoch, use_soft_label=0):
    fetches = [self.nlls_I2L_batchMean, self.nlls_L2I_train_bpd, self.nlls_L2I_test_bpd, self.consistent_loss,
               self.overall_cost_I2L, self.overall_cost_L2I]
    if args.oneside is None:
      fetches.append(self.Worker_I2L.model.update_ops)
      fetches.append(self.Worker_L2I.update_ops)
    elif args.oneside == 'I2L':
      fetches.append(self.Worker_I2L.model.update_ops)
    elif args.oneside == 'L2I':
      fetches.append(self.Worker_L2I.update_ops)
    else:
      raise Exception('Currently, only support None | I2L | L2I')

    feed_dict={
      # self.Worker_I2L.model.input_image: images.astype('float32'),
      # self.Worker_I2L.model.input_label: labels[:,None],
      self.Worker_I2L.model.lrn_rate: self.get_I2L_lr(),
      self.Worker_I2L.model.needImgAug: True,
      self.Worker_L2I.tf_lr: self.get_L2I_lr(),
      # self.image_LM: LMscores,
      self.trade_off_I2L: args.trade_off_I2L,
      self.trade_off_L2I: args.trade_off_L2I
    }

    splitted_image = np.split(images.astype('float32'), args.nr_gpu)
    splitted_label = np.split(labels, args.nr_gpu)
    splitted_LM = np.split(LMscores, args.nr_gpu)

    feed_dict.update({self.image_LM[i]: splitted_LM[i] for i in range(args.nr_gpu)})
    feed_dict.update({self.Worker_I2L.model.input_image[i]: splitted_image[i] for i in range(args.nr_gpu)})
    feed_dict.update({self.Worker_I2L.model.input_label[i]: splitted_label[i][:,None] for i in range(args.nr_gpu)})
    # Deal with xs and ys:
    x = np.cast[np.float32]((images - 127.5) / 127.5)
    x = np.split(x, self.Worker_L2I.args.nr_gpu)
    feed_dict.update({self.Worker_L2I.xs[i] : x[i] for i in range(self.Worker_L2I.args.nr_gpu)})
    if (use_soft_label == 2) or (use_soft_label == 1 and np.random.rand() < 0.8):
      soft_labels_ = self.sess.run(self.soft_labels, feed_dict={
        self.Worker_I2L.model.input_image: images.astype('float32'),
        self.Worker_I2L.model.needImgAug: True
      })
      if use_soft_label == 2:
        soft_labels_ -= 0.1
      feed_dict.update({self.Worker_L2I.hs[i]: soft_labels_ for i in range(self.Worker_L2I.args.nr_gpu)})
    else:
      y = np.split(labels, self.Worker_L2I.args.nr_gpu)
      feed_dict.update({self.Worker_L2I.ys[i] : y[i] for i in range(self.Worker_L2I.args.nr_gpu)})

    if args.oneside is None:
      nlls_I2L_mean, nlls_L2I_mean, nlls_L2I_mean_test, consistent_loss, overall_cost_I2L, overall_cost_L2I, _, _ = \
        self.sess.run(fetches, feed_dict)
    else:
      nlls_I2L_mean, nlls_L2I_mean, nlls_L2I_mean_test, consistent_loss, overall_cost_I2L, overall_cost_L2I, _, = \
        self.sess.run(fetches, feed_dict)
    if self.train_uidx % args.show_interval == (args.show_interval - 1):
      print('iter={}, I2L={}, L2I={}/{}, Consistent={}, Overall_I2L={}, Overall_L2I={}'.format(
        self.train_uidx, '{0:.4f}'.format(nlls_I2L_mean), '{0:.4f}'.format(nlls_L2I_mean),
        '{0:.4f}'.format(nlls_L2I_mean_test), '{0:.4f}'.format(consistent_loss), '{0:.4f}'.format(overall_cost_I2L),
        '{0:.4f}'.format(overall_cost_L2I)
      ))
    self.train_uidx += 1

  def data_dependent_init(self):
    global_init = tf.global_variables_initializer()
    _images, _labels, _ = train_data_iterator.next(self.Worker_L2I.args.init_batch_size)
    initializer_dict = {
      self.Worker_L2I.x_init: (np.cast[np.float32](_images) - 127.5)/127.5,
      self.Worker_L2I.y_init: _labels
    }
    train_data_iterator.reset()
    self.sess.run(global_init, initializer_dict)

  def L2I_TestNll(self, alpha_=1.):
    all_testnll = []
    for images, labels in test_data_iterator:
      feed_dict = {}
      x = np.cast[np.float32]((images - 127.5) / 127.5)
      x = np.split(x, args.nr_gpu)
      feed_dict.update({self.Worker_L2I.xs[i]: x[i] for i in range(args.nr_gpu)})
      if args.useSoftLabel == 1:
        soft_labels_ = self.sess.run(self.soft_labels, feed_dict={
          self.Worker_I2L.model.input_image: images.astype('float32'),
          self.Worker_I2L.model.needImgAug: False
        })
        one_hot_labels_ = np.zeros((args.batch_size, 10), dtype=np.float32)
        one_hot_labels_[np.arange(args.batch_size), labels] = 1.
        feed_dict.update({self.Worker_L2I.hs[i]: (1. - alpha_)*soft_labels_+alpha_*one_hot_labels_ for i in range(self.Worker_L2I.args.nr_gpu)})
      else:
        y = np.split(labels, args.nr_gpu)
        feed_dict.update({self.Worker_L2I.ys[i]: y[i] for i in range(args.nr_gpu)})
      all_testnll.append(self.sess.run([self.nlls_L2I_test_bpd], feed_dict))
    avg_testnll = np.mean(all_testnll)
    print('testnll=%f' % avg_testnll)

  def build_saver(self):
    self.saver = tf.train.Saver(max_to_keep=None)
    #tf.reset_default_graph()
    if args.load_params is not None:
      print('Reload from ', args.save_dir)
      self.saver.restore(self.sess, args.save_dir + '/' + args.load_params)
      print('Done')
    else:
      print('Start to initialize the two models')
      self.data_dependent_init()
      print('Done')

  def _steal_L2I(self):
    if args.steal_params_L2I is not None:
      # try to retrieve parameters NOT starting with "Variable" from a well-trained model
      success_ = 0
      import pickle
      with open(args.steal_params_L2I, 'rb') as f:
        old_model = pickle.load(f)
      for vidx, v in enumerate(tf.global_variables()):
        if v.name in old_model and not v.name.startswith('I2L/'):
          self.sess.run(self.assign_op(v, old_model[v.name][0]))
          success_ += 1
          print(vidx, len(tf.global_variables()))
      print('Retrieve %d / %d parameters from model %s' % (success_, len(old_model), args.steal_params_L2I))

      '''
      # this version can only reload "trainable vars"
      success_ = 0
      import pickle
      with open(args.steal_params_L2I, 'rb') as f:
        old_model = pickle.load(f)
      for vidx, v in enumerate(self.Worker_L2I.all_params):
        if v.name in old_model:
          self.sess.run(self.assign_op(v, old_model[v.name][0]))
          success_ += 1
          print(vidx, len(self.Worker_L2I.all_params))
      print('Retrieve %d / %d parameters from model %s' % (success_, len(old_model), args.steal_params_L2I))
      '''

  def _steal_I2L(self):
    if args.steal_params_I2L is not None:
      # try to retrieve parameters from a well-trained model
      success_ = 0
      import pickle
      with open(args.steal_params_I2L, 'rb') as f:
        old_model = pickle.load(f)
      for vidx, v in enumerate(self.Worker_I2L.model.all_variables):
        if v.name[4:] in old_model:
          self.sess.run(self.assign_op(v, old_model[v.name[4:]][0]))
          success_ += 1
          print(vidx, len(self.Worker_I2L.model.all_variables))
      print('Retrieve %d / %d parameters from model %s' % (success_, len(old_model), args.steal_params_I2L))

  def _reload_from_pkl(self, filename):
    success_ = 0
    import pickle

    with open(filename, 'rb') as f:
      old_model = pickle.load(f)

    for vidx, v in enumerate(self.Worker_I2L.model.all_variables):
      if v.name in old_model:
        self.sess.run(self.assign_op(v, old_model[v.name][0]))
        success_ += 1
        print(vidx, len(self.Worker_I2L.model.all_variables))

    for vidx, v in enumerate(self.Worker_L2I.all_params):
      if v.name in old_model:
        self.sess.run(self.assign_op(v, old_model[v.name][0]))
        success_ += 1
        print(vidx, len(tf.global_variables()))

    print('Retrieve %d / %d parameters from model ' % (success_, len(old_model)))

  def train(self):
    # do not delete the following three lines
    # self._reload_from_pkl('warm_values')
    # self.saver.save(self.sess, args.save_dir + '/params_stealt_models.ckpt')
    # return
    if args.load_params is None:
      self._steal_L2I()
      self._steal_I2L()
      self.saver.save(self.sess, args.save_dir + '/params_stealt_models.ckpt')
    for epoch in range(args.max_epochs):
      self.current_epoch = epoch
      for images, labels, LMscores in train_data_iterator:
        self.step(images, labels, LMscores, epoch, args.useSoftLabel)

      # if epoch % args.valid_interval == (args.valid_interval - 1):
      #  self.Worker_I2L.Valid(test_data_iterator, self.sess)
      #  self.L2I_TestNll()

      if epoch % args.save_interval == (args.save_interval - 1):
        self.saver.save(self.sess, args.save_dir  + '/params_' + str(epoch) + 'uidx' + str(self.train_uidx) + '.ckpt')
        self.Worker_L2I.Gen_Images(self.sess, self.current_epoch)

  def valid_I2L(self):
    self.Worker_I2L.Valid(test_data_iterator, self.sess)

  def valid_L2I(self):
    self.L2I_TestNll()
    '''
    for alpha_ in range(11):
      print('alpha=%f' % (alpha_ * 0.1))
      self.L2I_TestNll(alpha_ * 0.1)
    '''

  def valid_ImgGen(self):
    self.Worker_L2I.Gen_Images(self.sess, self.current_epoch)

  def dump_model_to_pkl(self):
    warm_models = {}
    print('Classifier')
    classifier_size = len(self.Worker_I2L.model.all_variables)
    for idx, v in enumerate(self.Worker_I2L.model.all_variables):
      vv = self.sess.run([v])
      warm_models[v.name] = vv
      if idx % 10 == 0:
        print('{}-{}'.format(idx, classifier_size))

    print('Generator')
    generator_size = len(self.Worker_L2I.all_params)
    for idx, v in enumerate(self.Worker_L2I.all_params):
      vv = self.sess.run([v])
      warm_models[v.name] = vv
      if idx % 10 == 0:
        print('{}-{}'.format(idx, generator_size))

    import pickle
    with open('warm_values', 'wb') as f:
        pickle.dump(warm_models, f, protocol=2)



def main(_):
  #L2Ipath='./pxpp_c_2.95/params_cifar.ckpt'
  monitor_ = monitor()
  monitor_.build_saver()

  if args.mode == 'train':
    monitor_.train()
  elif args.mode == 'I2L':
    monitor_.valid_I2L()
  elif args.mode == 'L2I':
    monitor_.valid_L2I()
  elif args.mode == 'ImgGen':
    monitor_.valid_ImgGen()
  else:
    print('Un supported mode: ' + args.mode)

if __name__ == '__main__':
  tf.app.run()
