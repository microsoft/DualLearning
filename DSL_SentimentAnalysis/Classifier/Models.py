"""
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

model for classification task
supports simple-rnn, lstm, hierarchical lstm
supports lstm with identity skip-connections(soft), parametric skip-connections(soft)
supports resnet, resnet with identity skip-connections(hard and soft), parametric skip connections(soft)
supports hybrid structure (lstm+resnet)
supports dropout on non-recurrent layers, gradient clipping, L2-regularization
"""
__author__ = 'v-yirwan'

import sys
import time

import numpy
import cPickle as pkl
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from Layers import get_layer
from Data import *
from Util import *

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_params(options):
    """
    Global (not LSTM) parameter. For the embedding and the classifier.
    """
    params = OrderedDict()
    # embedding
    if options['dataset'] != 'mnist':
        randn = rand_weight(options['n_words'], options['dim_word'])
        params['Wemb'] = randn.astype(config.floatX)

    # encoder layer
    params = get_layer(options['encoder'])[0](options, params,
                                                  prefix=options['encoder'])

    # classifier
    if options['lastHiddenLayer'] is not None:
        params['U'] = 0.01 * numpy.random.randn(options['lastHiddenLayer'],
                                                options['ydim']).astype(config.floatX)
        params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

        params['ToLastHidden_W'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                        options['lastHiddenLayer']).astype(config.floatX)
        params['ToLastHidden_b'] = numpy.zeros((options['lastHiddenLayer'],)).astype(config.floatX)


    else:
        params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                                options['ydim']).astype(config.floatX)
        params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def load_params(path, params):
    failer=0
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            failer += 1
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    print failer, ' failed out of ', len(params)
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def encoder_word_layer(tparams, state_below, options, mask=None):
    '''
    word(bottom)-level encoder for hierarchical architecture
    '''
    def _encode(x_sub, mask_sub, proj_sub):
        n_timesteps = x_sub.shape[0]
        n_samples = x_sub.shape[1]
        emb_sub = tparams['Wemb'][x_sub.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
        proj_sub = get_layer(options['encoder'])[1](tparams, emb_sub, options,
                                                    prefix=options['encoder']+'_word',
                                                    mask=mask_sub)
        return proj_sub[-1]
    proj_sub = tensor.alloc(numpy_floatX(0.), state_below.shape[2], options['dim_proj'])
    rval, update = theano.scan(_encode,
                               sequences=[state_below, mask],
                               outputs_info=[proj_sub],
                               name='word_encoder_layer',
                               n_steps=state_below.shape[0])
    return rval

def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    if options['dataset'] == 'mnist':
        print 'Using mnist dataset with single number input'
        x = tensor.matrix('x', dtype='float32')
    else:
        print 'Using text dataset with embedding input'
        x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # input word embedding
    if options['dataset'] == 'mnist':
        emb = x.reshape([n_timesteps, n_samples, options['dim_word']])
    else:
        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # dropout on embedding
    if options['dropout_input'] > 0:
        print 'Applying drop-out on input embedding (dropout_input:', options['dropout_input'], ')'
        emb = dropout_layer(emb, options['dropout_input'], use_noise, trng)

    # encoder information
    print 'Using', options['encoder'], 'unit'
    if options['truncate_grad'] is not None and options['truncate_grad'] > 0:
        print 'Using gradient truncation to', options['truncate_grad'], 'steps'
    else:
        options['truncate_grad'] = -1

    # encoding layer
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    # pooling
    if options['mean_pooling']:
        print 'Using mean_pooling'
        proj = (proj * mask[:, :, None]).sum(axis=0) # mean pooling
        proj = proj / mask.sum(axis=0)[:, None]
    else:
        print 'Using last hidden state'
        proj = proj[-1] # last hidden state

    sys.stdout.flush()

    # dropout on hidden states
    if options['lastHiddenLayer'] is not None:
        lastH = tensor.dot(proj, tparams['ToLastHidden_W']) + tparams['ToLastHidden_b']
        lastH = tensor.nnet.sigmoid(lastH)
        if options['dropout_output'] > 0:
            lastH = dropout_layer(lastH, options['dropout_output'], use_noise, trng)
        pred = tensor.nnet.softmax(tensor.dot(lastH, tparams['U']) + tparams['b'])
    else:
        if options['dropout_output'] > 0:
            print 'Applying drop-out on hidden states (dropout_output:', options['dropout_output'], ")"
            proj = dropout_layer(proj, options['dropout_output'], use_noise, trng)

        pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    # for training
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred') # sample by argmax

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    nlls = -tensor.log(pred[tensor.arange(n_samples), y] + off)

    return use_noise, x, mask, y, f_pred_prob, f_pred, nlls

class Model:
    def __init__(self,
                 dim_word=500, # word embeding dimension
                 dim_proj=1024,  # LSTM number of hidden units
                 patience=10,  # Number of epoch to wait before early stop if no progress
                 max_epochs=5000,  # The maximum number of epoch to run
                 decay_c=-1.,  # Weight decay (for L2-regularization)
                 clip_c=-1., # gradient clipping threshold
                 lrate=1.,  # Learning rate for sgd (not used for adadelta and rmsprop)
                 n_words=10000,  # Vocabulary size
                 optimizer='adadelta',
                 encoder='lstm', # name of encoder unit, refer to 'layers'
                 encoder2=None, # only used in hybrid mode
                 hierarchical=False, # whether use hierarchical structure
                 hier_len=None, # length of bottom (word-level) encoder
                 hybrid=False, # whether use hybrid model
                 mean_pooling=False, # use last hidden state if false
                 unit_depth=-1, # recurrent depth of residual unit
                 skip_steps=-1, # skip connection length (h(t) -> h(t+skip_steps))
                 skip_steps2=-1, # only used in hybrid mode
                 truncate_grad=-1, # e number of steps to use in truncated BPTT, set to -1 if not to apply
                 saveto='model.npz',  # The best model will be saved there
                 dispFreq=50,  # Display the training progress after this number of updates
                 validFreq=300,  # Compute the validation error after this number of updates
                 newDumpFreq=5000000, # Dump model into a new file after this number of updates
                 maxlen=None,  # Sequence longer then this get ignored
                 batch_size=16,  # The batch size during training.
                 batch_len_threshold=None, # use dynamic batch size if sequence lengths exceed this threshold
                 valid_batch_size=16,  # The batch size used for validation/test set.
                 dataset='text', # dataset dype
                 corpus='imdb.pkl', # path to load training data
                 start_iter=0,
                 start_epoch=0,
                 noise_std=0.,
                 lastHiddenLayer=None,
                 dropout_output=None,  # Dropout on output hidden states (before softmax layer)
                 dropout_input=None, # Dropout on input embeddings
                 reload_options=None, # Path to a saved model options we want to start from
                 reload_model=None,  # Path to a saved model we want to start from.
                 embedding=None, # Path to the word embedding file (otherwise randomized)
                 warm_LM=None,
                 test_size=None,  # If >0, we keep only this number of test example.
                 monitor_grad=False, # Print gradient norm to log file at each iteration if set True
                 logFile='log.txt' # Path to log file
                 ):

        # Model options
        self.model_options = locals().copy()
        self.model_options['self'] = None

        # log files
        self.F_log = open(logFile, "a")

        if start_iter == 0:
            self.F_log.write("model options:\n")
            for kk, vv in self.model_options.iteritems():
                self.F_log.write("\t"+kk+":\t"+str(vv)+"\n")
            self.F_log.write("-----------------------------------------\n")

        pkl.dump(self.model_options, open('%s.pkl' % saveto, 'wb'))

        print 'Loading data...',
        if dataset == 'mnist':
            self.trainSet, self.validSet, self.testSet = load_mnist(path=corpus,
                                                                    fixed_permute=True,
                                                                    rand_permute=False)
        else:
            self.trainSet, self.validSet, self.testSet = load_data(path=corpus,
                                                                   n_words=n_words,
                                                                   maxlen=maxlen,
                                                                   sort_by_len=True,
                                                                   fixed_valid=True)
        print 'Done! '
        print 'Training', len(self.trainSet[0]), 'Valid', len(self.validSet[0]), 'Test', len(self.testSet[0])
        sys.stdout.flush()

        if test_size > 0:
            test_size = min(test_size, len(self.testSet[0]))
            idx = numpy.arange(len(self.testSet[0]))
            numpy.random.shuffle(idx)
            idx = idx[:test_size]
            self.testSet = ([self.testSet[0][n] for n in idx], [self.testSet[1][n] for n in idx])

        # number of classes
        ydim = numpy.max(self.trainSet[1]) + 1
        self.model_options['ydim'] = ydim

        print 'Initializing model parameters...',
        params = init_params(self.model_options)
        print 'Done'
        print 'Model size:', self.model_options['dim_word'], '*', self.model_options['dim_proj']
        sys.stdout.flush()

        # load pre-trained word embedding
        if embedding is not None and os.path.exists(embedding):
            Wemb = numpy.array(numpy.load(open(embedding, "rb")))
            if Wemb.shape[0] == self.model_options['n_words'] and \
                            Wemb.shape[1] == self.model_options['dim_word']:
                print 'Using pre-trained word embedding'
                params['Wemb'] = Wemb.astype(numpy.float32) # bug fixed
                print 'vocab size', params['Wemb'].shape[0], ', dim', params['Wemb'].shape[1]

        # reload options
        if reload_options is not None and os.path.exists(reload_options):
            print "Reloading model options...",
            with open(reload_options, 'rb') as f:
                self.model_options = pkl.load(f)
            print "Done"

        # reload parameters
        self.start_iter = 0
        self.start_epoch = 0
        self.history_errs = []
        if reload_model is not None and os.path.exists(reload_model): # bug fixed
            print 'Reloading model parameters...',
            load_params(reload_model, params)
            self.start_iter = start_iter
            self.start_epoch = start_epoch
            #self.history_errs = list(numpy.load(self.model_options['reload_model'])['history_errs'])
            print 'Done'
        sys.stdout.flush()

        if warm_LM is not None:
            print 'Steal from language model'
            warmLM_ = numpy.load(warm_LM)
            assert params['lstm_W'].shape == warmLM_['encoder_W'].shape
            assert params['lstm_b'].shape == warmLM_['encoder_b'].shape
            assert params['lstm_U'].shape == warmLM_['encoder_U'].shape
            assert params['Wemb'].shape == warmLM_['Wemb'].shape
            params['lstm_W'] = warmLM_['encoder_W']
            params['lstm_b'] = warmLM_['encoder_b']
            params['lstm_U'] = warmLM_['encoder_U']
            params['Wemb'] = warmLM_['Wemb']

        self.tparams = init_tparams(params)

        # build model
        mask_proj = None
        # vanilla structure
    def GetNll(self):
        print 'Using vanilla structure'
        self.use_noise, x, mask, y, \
        self.f_pred_prob, self.f_pred, nlls = \
            build_model(self.tparams, self.model_options)
        #inps = [x, mask, y]
        return x, mask, y, nlls

    def get_accu(self, data, iterator, hier_len=None):
        """
        Just compute the error
        modified to support hierarchical mode
        """
        valid_acc = 0
        for _, valid_index in iterator:
            if hier_len is not None:
                x, mask, mask_proj, y = prepare_data_hier([data[0][t] for t in valid_index],
                                                           numpy.array(data[1])[valid_index],
                                                           hier_len=hier_len)
                preds = self.f_pred(x, mask, mask_proj)
            else:
                x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                          numpy.array(data[1])[valid_index],
                                          maxlen=None,
                                          dataset=self.model_options['dataset'])
                preds = self.f_pred(x, mask) # result obtained by argmax
            valid_acc += (preds == y).sum() # note that batch is sorted in hier-mode
        valid_acc = numpy_floatX(valid_acc) / numpy_floatX(len(data[0])) # accuracy

        return valid_acc

    def save_model(self, savefile, best_p=None):
        if best_p is not None: # save the best model so far
            params = best_p
        else:
            params = unzip(self.tparams)
        numpy.savez(savefile, history_errs=self.history_errs, **params)
        pkl.dump(self.model_options, open('%s.pkl' % self.model_options['saveto'], 'wb'))

    def valid(self):
        train_acc = self.get_accu(self.trainSet, self.kf_train)
        #hier_len=self.model_options['hier_len'])
        valid_acc = self.get_accu(self.validSet, self.kf_valid)
                                  #hier_len=self.model_options['hier_len'])
        test_acc = self.get_accu(self.testSet, self.kf_test)
                                 #hier_len=self.model_options['hier_len'])
        return train_acc, valid_acc, test_acc

    def evaluate(self, *dataset):
        acc = []
        for k in xrange(len(dataset)):
            data = dataset[k]
            idx = get_minibatches_idx(len(data[0]), 16)
            acc.append(self.get_accu(data, idx))
        return acc



if __name__ == '__main__':
    pass




