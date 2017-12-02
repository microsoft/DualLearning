# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import sys
import os

import cifar_input
import numpy as np
import resnet_model_basic as resnet_model
import tensorflow as tf
import data.cifar10_data as cifar10_data



def lr_I2L(train_step):
  #step_wise = [40000,60000,80000] # this is the one for original
  step_wise = [51000,76000,102000]
  if train_step < step_wise[0]:
    return 0.1
  elif train_step < step_wise[1]:
    return 0.01
  elif train_step < step_wise[2]:
    return 0.001
  else:
    return 0.0001

class worker_I2L(object):
  def __init__(self, args):

    hps = resnet_model.HParams(batch_size=args.batch_size,
                               num_classes=10,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=18,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')
    self.args = args
    self.model = resnet_model.ResNet(hps, args.mode, use_wide_resnet=args.use_wide_resnet, nr_gpu=args.nr_gpu)
    self.model.build_graph()

    truth = tf.argmax(tf.concat(self.model.labels, axis=0), axis=1)
    predictions = tf.argmax(tf.concat(self.model.predictions,axis=0), axis=1)
    self.right_decision = tf.reduce_sum(tf.to_float(tf.equal(predictions, truth)))

  def GetLoss(self):
    return self.model.nlls, self.model.GetWeightDecay()

  def Valid(self, test_data, sess):
    with tf.device('/gpu:0'):
      cost_all = self.model.nlls[0]
      for i in range(1, self.args.nr_gpu):
        cost_all += self.model.nlls[i]

    m_sample = 0
    m_correct = 0.
    costs = 0.
    for test_image, test_label in test_data:
      m_sample += test_image.shape[0]

      splitted_image = np.split(test_image.astype('float32'), self.args.nr_gpu)
      splitted_label = np.split(test_label, self.args.nr_gpu)

      feed_dict = {self.model.needImgAug: False}
      feed_dict.update({self.model.input_image[i]: splitted_image[i] for i in range(self.args.nr_gpu)})
      feed_dict.update({self.model.input_label[i]: splitted_label[i][:, None] for i in range(self.args.nr_gpu)})

      _cost, _right_decision = sess.run([cost_all, self.right_decision], feed_dict)
      costs += np.sum(_cost)
      m_correct += _right_decision
    test_loss = costs / m_sample
    test_acc = m_correct * 1. / m_sample
    print('[I2L] test_nll={},test_acc={}'.format(
        '{0:.4f}'.format(test_loss), '{0:.6f}'.format(test_acc) )
    )
