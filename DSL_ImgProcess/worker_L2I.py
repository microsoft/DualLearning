# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data

class worker_L2I(object):
    def __init__(self, args, num_labels, image_shape):
        # Default parameters
        self.num_labels = num_labels
        self.image_shape=image_shape
        self.args = args

        # Data userd for data-dependent parameter initialization
        self.x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + self.image_shape)
        self.xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + self.image_shape) for _ in range(args.nr_gpu)]
        self.y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
        self.h_init = tf.one_hot(self.y_init, self.num_labels)

        # parameters used for sampling
        self.y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), self.num_labels), args.nr_gpu)
        # self.h_sample = [tf.one_hot(tf.Variable(self.y_sample[i], trainable=False), self.num_labels) for i in range(args.nr_gpu)]
        # the above line is the version used for icml paper. I revise it as follows
        self.h_sample = [tf.one_hot(self.y_sample[i], self.num_labels) for i in range(args.nr_gpu)]
        self.ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
        self.hs = [tf.one_hot(self.ys[i], self.num_labels) for i in range(args.nr_gpu)]
        # create the model
        self.model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity }
        self.model = tf.make_template('model', model_spec)

        # run once for data dependent initialization of parameters
        # in the original code, it is " gen_par = self.model(...)"; when init=True, it will run initilization automatically
        self.model(self.x_init, self.h_init, init=True, dropout_p=args.dropout_p, **self.model_opt)

        # keep track of moving average
        self.all_params = tf.trainable_variables()
        self.ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
        self.maintain_averages_op = tf.group(self.ema.apply(self.all_params))

        # parameters for optimization
        self.tf_lr = tf.placeholder(tf.float32, shape=())

    def GetLoss(self):
        # get loss gradients over multiple GPUs
        loss_gen = []
        loss_gen_test = []
        for i in range(self.args.nr_gpu):
            with tf.device('/gpu:%d' % i):
                # train
                gen_par = self.model(self.xs[i], self.hs[i], ema=None, dropout_p=self.args.dropout_p, **self.model_opt)
                loss_gen.append(nn.discretized_mix_logistic_loss(self.xs[i], gen_par, sum_all=False))

                # test
                gen_par = self.model(self.xs[i], self.hs[i], ema=self.ema, dropout_p=0., **self.model_opt)
                loss_gen_test.append(nn.discretized_mix_logistic_loss(self.xs[i], gen_par))

        return loss_gen, loss_gen_test

    def GetOverallLoss(self):
        # get loss gradients over multiple GPUs
        loss_gen = []
        loss_gen_test = []
        for i in range(self.args.nr_gpu):
            with tf.device('/gpu:%d' % i):
                # train
                gen_par = self.model(self.xs[i], self.hs[i], ema=None, dropout_p=self.args.dropout_p, **self.model_opt)
                loss_gen.append(nn.discretized_mix_logistic_loss(self.xs[i], gen_par, sum_all=False))

                # test
                gen_par = self.model(self.xs[i], self.hs[i], ema=self.ema, dropout_p=0., **self.model_opt)
                loss_gen_test.append(nn.discretized_mix_logistic_loss(self.xs[i], gen_par))

        # add the lossx to /gpu:0
        with tf.device('/gpu:0'):
            for i in range(1,self.args.nr_gpu):
                loss_gen[0] += loss_gen[i]
                loss_gen_test[0] += loss_gen_test[i]

            # training op
            #optimizer = tf.group(nn.adam_updates(self.all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995), self.maintain_averages_op)

        # convert loss to bits/dim
        self.bits_per_dim = loss_gen[0]/(self.args.nr_gpu*np.log(2.)*np.prod(self.image_shape)*self.args.batch_size)
        self.bits_per_dim_test = loss_gen_test[0]/(self.args.nr_gpu*np.log(2.)*np.prod(self.image_shape)*self.args.batch_size)

    def Update(self, grads, useSGD=False):
        if useSGD:
            print('Use pure SGD for Label-->Image tasks')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.tf_lr)
            apply_op = optimizer.apply_gradients(zip(grads, self.all_params))
            self.update_ops = tf.group(apply_op)
        else:
            self.update_ops = tf.group(nn.adam_updates(self.all_params, grads, lr=self.tf_lr, mom1=0.95, mom2=0.9995), self.maintain_averages_op)

    def build_sample_from_model(self):
        # sample from the model
        self.new_x_gen = []
        for i in range(self.args.nr_gpu):
            with tf.device('/gpu:%d' % i):
                gen_par = self.model(self.xs[i], self.h_sample[i], ema=self.ema, dropout_p=0, **self.model_opt)
                self.new_x_gen.append(nn.sample_from_discretized_mix_logistic(gen_par, self.args.nr_logistic_mix))

    def _sample_from_model(self, sess):
        x_gen = [np.zeros((self.args.batch_size,) + self.image_shape, dtype=np.float32) for _ in range(self.args.nr_gpu)]
        for yi in range(self.image_shape[0]):
            for xi in range(self.image_shape[1]):
                new_x_gen_np = sess.run(self.new_x_gen, {self.xs[i]: x_gen[i] for i in range(self.args.nr_gpu)})
                for i in range(self.args.nr_gpu):
                    x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
        return np.concatenate(x_gen, axis=0)


    def Gen_Images(self, sess, epoch):
        sample_x = self._sample_from_model(sess)
        img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(self.args.batch_size*self.args.nr_gpu))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title=self.args.data_set + ' samples')
        plotting.plt.savefig(os.path.join(self.args.save_dir,'%s_sample%d.png' % (self.args.data_set, epoch)))
        plotting.plt.close('all')
