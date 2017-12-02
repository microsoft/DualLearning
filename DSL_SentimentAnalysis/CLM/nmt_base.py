'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

# Added by v-yixia
import bottleneck, random
from multiprocessing import Process, Queue, Lock
#import nltk.translate as nt
sys.setrecursionlimit(10000)

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

def itemlist_NoEmb(tparams):
    return [vv for kk, vv in tparams.iteritems() if 'Wemb' not in kk]

# dropout
'''
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj
'''

def dropout_layer(state_before, use_noise, trng, dropout_rate = 0.5):
    proj = tensor.switch(
            use_noise,
            (state_before * trng.binomial(state_before.shape,
                                          p=(1. - dropout_rate), n=1,
                                          dtype=state_before.dtype)),
            state_before * (1. - dropout_rate))
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer')
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask

# batch preparation
def prepare_data_x(seqs_x, maxlen=None, pad_eos = True, pad_sos=False, n_word=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x


        if len(lengths_x) < 1:
            return None, None,

    n_samples = len(seqs_x)
    if pad_eos:
        maxlen_x = numpy.max(lengths_x) + 1
    else:
        maxlen_x = numpy.max(lengths_x)


    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')

    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')

    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        if pad_eos:
            x_mask[:lengths_x[idx]+1, idx] = 1.
        else:
            x_mask[:lengths_x[idx], idx] = 1.

    if pad_sos:
        x = numpy.concatenate((
            numpy.full([1, n_samples], n_word - 1, dtype='int64'), x
        ), axis=0)
        x_mask = numpy.concatenate((
            numpy.full([1, n_samples], 1., dtype='float32'), x_mask
        ), axis=0)

    return x, x_mask

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval

# LSTM layer for LM
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None, m_label=2):
    """
    Init the LSTM parameter
    """
    if nin is None:
        nin = options['dim_word']
    if dim is None:
        dim = options['dim_proj']

    params['Wemb_ctx'] = norm_weight(m_label, nin)
    params[_p(prefix, 'ctx2hidden')] = norm_weight(nin, 4 * dim)

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W

    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * dim,))
    params[_p(prefix, 'b')] = b.astype('float32')

    return params

def lstm_layer(tparams, state_below, ctx_emb, options, prefix='lstm', mask=None,
               one_step=False, init_state=None, cell_state=None, **kwargs):
    if one_step:
        assert init_state, 'previous hidden state must be provided'
        assert cell_state, 'previous cell state must be provided'

    # I assume state_below is (seq_len, batch_size, 4 dim)
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    dim = options['dim_proj']

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1) # bug fixed: no mask for sampler

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_): # m_ (batch_size, ) | x_ (batch_size, 4 dim) | h_/c_ (batch_size, dim)
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim)) # block input
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')] +\
                  tensor.dot(ctx_emb, tparams[_p(prefix, 'ctx2hidden')])

    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)
    if cell_state is None:
        cell_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:
        rval = _step(mask, state_below, init_state, cell_state)
    else:
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[init_state, cell_state],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps)
    return rval

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_label',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params

def init_lm_params(options):
    params = OrderedDict()

    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    # x: row embedding; y: column embedding

    params = get_layer(options['encoder'])[0](options, params,
            prefix='encoder',
            nin=options['dim_word'],
            dim=options['dim_proj'])
    params = get_layer('ff')[0](options, params,
            prefix='ff_logit_lstm',
            nin=options['dim_proj'],
            nout=options['dim_word'],
            ortho=False)

    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev', nin=options['dim_word'], nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_label', nin=options['dim_word'], nout=options['dim_word'], ortho=False)

    params = get_layer('ff')[0](options, params,
            prefix='ff_logit',
            nin=options['dim_word'],
            nout=options['n_words'])

    return params

# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next

# build the function of LM-Reward
def build_lm_scorer(tparams, options):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    srcx = tensor.matrix('srcx', dtype='int64')
    srcx_mask = tensor.matrix('srcx_mask', dtype='float32')

    x = srcx[:-1, :]

    n_timesteps = x.shape[0]
    n_samples   = x.shape[1]

    emb  = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])

    if options['dropout_input'] is not None and options['dropout_input'] > 0:
        emb = dropout_layer(emb, use_noise, trng, options['dropout_input'])

    # input initial states
    init_state = tensor.alloc(0., n_samples, options['dim_proj']) #tensor.matrix('init_state', dtype='float32')
    init_cell  = tensor.alloc(0., n_samples, options['dim_proj']) #tensor.matrix('init_cell', dtype='float32')

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
            prefix='encoder',
            init_state=init_state,
            cell_state=init_cell)


    proj_h = proj[0] # all hidden states

    #next_states = [st[-1] for st in proj] # first last hidden_state, second last cell_state

    if options['dropout_output'] is not None and options['dropout_output'] > 0:
        proj_h = dropout_layer(proj_h, use_noise, trng, options['dropout_output'])


    # compute word probabilities
    def _prob(proj_h):
        logit_lstm = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
        logit_prev = get_layer('ff')[1](tparams, emb, options, prefix='ff_logit_prev', activ='linear')
        logit = tensor.tanh(logit_lstm + logit_prev)

        #logit = tensor.tanh(logit_lstm)
        # split to calculate
        logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape # n_timesteps * n_samples * n_words
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))
        return probs

    probs = _prob(proj_h)

    # cost
    y = srcx[1:,:]
    y_flat =  y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat

    # probs:(seq,batch,worddim) <-> x:(seq,batch) become the right place value
    # y:(seq_len, batch_size)
    def _cost(probs):
        cost = -tensor.log(probs.flatten()[y_flat_idx] + 1e-10)
        cost = cost.reshape([y.shape[0], y.shape[1]])
        # cost = cost.mean(0)
        cost = (cost * srcx_mask[1:,:]).sum(0)
        return cost

    cost = _cost(probs)

    return trng, use_noise, srcx, srcx_mask, cost

# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + lr * ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, inps, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inps, cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update

def sgd_(lr, tparams, grads):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=gsup + pup, profile=profile)

    return f_update

def sgd_FreezeEmb(lr, tparams, grads, inps, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems() if 'Wemb' not in k]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inps, cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist_NoEmb(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update
    
'''
Added by v-yixia
The following functions are used for sampling examples
'''

# only sample with fixed length
def preprocess_data(seqs_x, seqs_y, minlen, maxlen):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    fixed_seqs_x, fixed_seqs_y, tosample_seqs_x, tosample_seqs_y, tosample_len_x, tosample_len_y = [], [], [], [], [], []

    for l_x, l_y, s_x, s_y in zip(lengths_x, lengths_y, seqs_x, seqs_y):
        if l_y < maxlen and l_y > minlen:
            tosample_seqs_x.append(s_x)
            tosample_seqs_y.append(s_y)
            tosample_len_x.append(l_x)
            tosample_len_y.append(l_y)
        else:
            fixed_seqs_x.append(s_x)
            fixed_seqs_y.append(s_y)

    if len(tosample_len_y) < 1:
        return None, None, None, None, fixed_seqs_x, fixed_seqs_y

    n_samples = len(tosample_seqs_y)
    maxlen_x = numpy.max(tosample_len_x) + 1

    # pad batches and create masks
    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(tosample_seqs_x):
        x[:tosample_len_x[idx], idx] = s_x
        x_mask[:tosample_len_x[idx]+1, idx] = 1.

    return x, x_mask, tosample_seqs_x, tosample_seqs_y, fixed_seqs_x, fixed_seqs_y

def postprocess_data(seqs_x, seqs_y, weights, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        new_weights = []
        for l_x, s_x, l_y, s_y, ww in zip(lengths_x, seqs_x, lengths_y, seqs_y, weights):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_weights.append(ww)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        weights = new_weights

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask, numpy.array(weights, dtype = numpy.float32)[None]

def build_batch_sampler(tparams, options, trng, use_noise):
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype = 'float32')
    xr = x[::-1]
    xr_mask = x_mask[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder', mask = x_mask)
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r', mask = xr_mask)

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    # ctx_mean = ctx.mean(0) # This is the original code
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init in batch mode...',
    outs = [init_state, ctx] # original code, have bug
    # outs = [init_state, ctx_mean]  # bug fixed
    f_init = theano.function([x, x_mask], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    x_extend_mask = tensor.matrix('x_extend_mask', dtype = 'float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., y.size, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            context_mask = x_extend_mask,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next in batch mode..',
    inps = [y, ctx, x_extend_mask, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next

def beam_search_sample(f_init, f_next, x, x_mask, gpulock, k=16, maxlen=50):
    '''
    Only used for Batch Beam Search;
    Do not Support Stochastic Sampling
    '''
    # _ = os.system('cls')

    batch_size = x.shape[1]
    sample = [ [] for _ in xrange(batch_size) ]
    sample_score = [ [] for _ in xrange(batch_size) ]


    lives_k = [1] * batch_size
    deads_k = [0] * batch_size

    batch_hyp_samples = [ [[]] for _ in xrange(batch_size)  ]
    batch_hyp_scores  = [ numpy.zeros(ii, dtype = numpy.float32) for ii in lives_k]

    # get initial state of decoder rnn and encoder context
    if gpulock is not None:
        gpulock.acquire()
    ret = f_init(x, x_mask)
    if gpulock is not None:
        gpulock.release()

    next_state, ctx0 = ret[0], ret[1]
    next_w = numpy.array([-1] * batch_size).astype('int64')  # indicator for the first target word (bos target)


    for ii in xrange(maxlen):
        ctx = numpy.zeros([ctx0.shape[0], numpy.array(lives_k).sum(), ctx0.shape[2]], dtype = numpy.float32)
        x_extend_masks = numpy.zeros( [x_mask.shape[0], numpy.array(lives_k).sum()], dtype = numpy.float32 )
        cursor_start, cursor_end = 0, lives_k[0]
        for jj in xrange(batch_size):
            if lives_k[jj] > 0:
                ctx[:, cursor_start : cursor_end, :] = numpy.repeat(ctx0[:, jj, :][:, None, :], lives_k[jj], axis = 1)
                x_extend_masks[:, cursor_start : cursor_end] = numpy.repeat(x_mask[:, jj][:, None], lives_k[jj], axis = 1)
            if jj < batch_size - 1:
                cursor_start = cursor_end
                cursor_end += lives_k[jj + 1]

        inps = [next_w, ctx, x_extend_masks, next_state]
        if gpulock is not None:
            gpulock.acquire()
        ret = f_next(*inps)
        if gpulock is not None:
            gpulock.release()
        next_w_list = []
        next_state_list = []

        next_p, next_state = ret[0], ret[2]
        cursor_start, cursor_end = 0, lives_k[0]

        for jj in xrange(batch_size):
            if cursor_start == cursor_end:
                if jj < batch_size - 1:
                    cursor_end += lives_k[jj + 1]
                continue
            index_range = range(cursor_start, cursor_end)
            cand_scores = batch_hyp_scores[jj][:, None] - numpy.log(next_p[index_range, :])
            cand_flat = cand_scores.flatten()

            # ranks_flat = cand_flat.argsort()[:(k - deads_k[jj])]
            # ranks_flat = numpy.array([zz[0] for zz in heapq.nsmallest(k - deads_k[jj], enumerate(cand_flat), key = lambda ele: ele[1])])
            # ranks_flat = numpy.array(map(list(cand_flat).index, heapq.nsmallest(k - deads_k[jj], cand_flat)))
            ranks_flat = bottleneck.argpartsort(cand_flat, n = k - deads_k[jj])[:k - deads_k[jj]]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k - deads_k[jj]).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(batch_hyp_samples[jj][ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[cursor_start + ti]))

            # check the finished samples
            new_live_k = 0
            batch_hyp_samples[jj] = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample[jj].append(new_hyp_samples[idx])
                    sample_score[jj].append(new_hyp_scores[idx])
                    deads_k[jj] += 1
                else:
                    new_live_k += 1
                    batch_hyp_samples[jj].append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])

            batch_hyp_scores[jj] = numpy.array(hyp_scores)
            lives_k[jj] = new_live_k

            if jj < batch_size - 1:
                cursor_start = cursor_end
                cursor_end += lives_k[jj + 1]

            if hyp_states:
                next_w_list += [ w[-1] for w in batch_hyp_samples[jj] ]
                next_state_list.append(hyp_states)

        if numpy.array(lives_k).sum() > 0:
            next_w = numpy.array(next_w_list)
            next_state = numpy.row_stack(next_state_list[:])
        else:
            break


    # dump every remaining one
    for jj in xrange(batch_size):
        if lives_k[jj] > 0:
            for idx in xrange(lives_k[jj]):
                sample[jj].append(batch_hyp_samples[jj][idx])
                sample_score[jj].append(batch_hyp_scores[jj][idx])

    return sample, sample_score
'''
def batch_gensample_model(in_queue, out_queue, f_init, f_next, k, gpulock, n_resample, minlen, maxlen, bleulowerbound = 0.25):
    while True:
        req = in_queue.get()
        if req is None:
            break
        x, x_mask, tosample_seqs_x, tosample_seqs_y, fixed_seqs_x, fixed_seqs_y = preprocess_data(req[1], req[2], minlen, maxlen)
        XX, YY = fixed_seqs_x, fixed_seqs_y
        assert len(XX) == len(YY), 'len(XX) NE len(YY)'
        WW = [1.] * len(XX)
        resample_counter = 0
        if x is not None:
            samples, scores = beam_search_sample(f_init, f_next, x, x_mask, gpulock, k, maxlen)
            resample_size = len(tosample_seqs_x)
            for idx in xrange(resample_size):
                bleu_scores = []
                str_tt = [str(y) for y in tosample_seqs_y]
                for ii in xrange(len(samples[idx])):
                    str_ss = [str(x) for x in samples[idx][ii]]
                    bleu_scores.append( nt.bleu([str_tt], str_ss, [0.25] * 4) )
                sorted_id = bottleneck.argpartsort(-numpy.array(bleu_scores, dtype = 'float32'), n = n_resample)[:n_resample]

                start_idx = 0

                if bleu_scores[sorted_id[0]] < 0.999:
                    XX.append(tosample_seqs_x[idx])
                    YY.append(tosample_seqs_y[idx])
                    WW.append(1.)
                    start_idx = 1

                for ii in xrange(start_idx, n_resample):
                    if bleu_scores[sorted_id[ii]] < bleulowerbound:
                        break
                    XX.append(tosample_seqs_x[idx])
                    YY.append(samples[idx][sorted_id[ii]])
                    WW.append(bleu_scores[sorted_id[ii]])
                    resample_counter += 1

        out_queue.put((req[0], XX, YY, WW, resample_counter))
    out_queue.put(None)
    print 'Process %d Finish One' % os.getpid()
    return
'''
def batch_gensample(trainX, trainY, f_init_batch, f_next_batch, block_size, k, n_process, n_resample, minlen, maxlen):
    in_queue = Queue()
    out_queue = Queue()

    # Fill the task queue
    def _FillQueue():
        idx = 0
        while True:
            xx = trainX[idx * block_size : (idx + 1) * block_size]
            yy = trainY[idx * block_size : (idx + 1) * block_size]
            in_queue.put((idx, xx, yy))
            if (idx + 1) * block_size >= len(trainX):
                break
            idx += 1

        for _ in xrange(n_process):
            in_queue.put(None)
        return

    def _collect_tasks():
        XX = []
        YY = []
        WW = []
        resample_counter = 0
        cnt = 0
        while cnt < n_process:
            ret = out_queue.get()
            if ret is None:
                cnt += 1
            else:
                XX += ret[1]
                YY += ret[2]
                WW += ret[3]
                resample_counter += ret[4]

        return XX, YY, WW, resample_counter

    _FillQueue()
    # batch_translate_model(in_queue, out_queue, f_init, f_next, k, options, 101)

    processes = [None] * n_process
    gpulock = Lock()
    for midx in xrange(n_process):
        processes[midx] = Process(target=batch_gensample_model,
                                    args=(in_queue, out_queue, f_init_batch, f_next_batch, k, gpulock, n_resample, minlen, maxlen))
        processes[midx].start()


    return _collect_tasks()

def brute_add_weight(trainX, trainY, n_resample, select_range):
    assert len(trainX) == len(trainY), 'len(trainX) NE len(trainY)'
    weights = [n_resample * 1. if len(x) >= select_range[0] and len(x) < select_range[1] else 1. for x in trainX]
    return trainX, trainY, weights

def duplicate_data(trainX, trainY, n_resample, dup_range):
    datasize = len(trainX)
    assert datasize == len(trainY), 'trainX and trainY are of different size'

    for idx in xrange(datasize):
        x = trainX[idx]
        y = trainY[idx]
        if len(x) >= dup_range[0] and len(x) <= dup_range[1] and len(y) >= dup_range[0] and len(y) <= dup_range[1]:
            trainX += ([x] * n_resample)
            trainY += ([y] * n_resample)


    combined = zip(trainX, trainY)
    random.shuffle(combined)
    shuf_sampleX, shuf_sampleY = zip(*combined)

    return shuf_sampleX, shuf_sampleY, [1.] * len(shuf_sampleX)

def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          reload_=False,
          overwrite=False,
          n_resample = 4,
          beam_search_size = 16,
          resample_minlen = 20,
          resample_maxlen = 30,
          flag_freeze_embed = False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    print 'Open Log File'
    fp_log = open('Log_' + time.strftime('%m-%d-%H-%M-%S'), 'w')
    fp_log.write('Start Time = ' + time.strftime('%c'))
    n_process = 2
    batch_load_size = 10
    train = TextIterator(datasets[0], datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size * batch_load_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)
    f_init_batch, f_next_batch = build_batch_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    _weights = tensor.matrix('_weights', dtype = 'float32')
    inps2 = [x, x_mask, y, y_mask, _weights]

    cost *= (_weights)
    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps2, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    if flag_freeze_embed:
        grads = tensor.grad(cost, wrt=itemlist_NoEmb(tparams))
    else:
        grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    if flag_freeze_embed:
        f_grad_shared, f_update = eval(optimizer + '_FreezeEmb')(lr, tparams, grads, inps2, cost)
    else:    
        f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps2, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            use_noise.set_value(1.)
            # XX, YY, WW, resample_counter = batch_gensample(x, y, f_init_batch, f_next_batch, batch_size, beam_search_size, n_process, n_resample, resample_minlen, resample_maxlen) # ????
            # x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen, n_words_src=n_words_src, n_words=n_words)
            # XX, YY, WW = brute_add_weight(x, y, n_resample, [resample_minlen, resample_maxlen])
            XX, YY, WW = duplicate_data(x, y, n_resample, dup_range = [resample_minlen, resample_maxlen])
            batch_idx, n_prepared_data = 0, len(XX)
            # print 'Resample Counter = %d' % resample_counter
            # fp_log.write('Resample Counter = %d\n' % resample_counter)

            while batch_idx * batch_size < n_prepared_data:
                x, x_mask, y, y_mask, weights = postprocess_data(XX[batch_idx * batch_size : (batch_idx + 1) * batch_size],
                                                                 YY[batch_idx * batch_size : (batch_idx + 1) * batch_size],
                                                                 WW[batch_idx * batch_size : (batch_idx + 1) * batch_size], maxlen=None)
                batch_idx += 1
                uidx += 1

                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    uidx -= 1
                    continue

                ud_start = time.time()

                # compute cost, grads and copy grads to shared variables
                # cost = f_grad_shared(x, x_mask, y, y_mask) # original code
                cost = f_grad_shared(x, x_mask, y, y_mask, weights)

                # do the update on parameters
                f_update(lrate)

                ud = time.time() - ud_start

                # check for bad numbers, usually we remove non-finite elements
                # and continue training - but not done here
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN or Inf detected'
                    fp_log.write('NaN or Inf detected\n')
                    return 1., 1., 1.

                # verbose
                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
                    fp_log.write('Epoch '+str(eidx)+'\tUpdate '+str(uidx)+'\tCost '+str(cost)+'\tUD '+str(ud)+'\n')

                # save the best model so far, in addition, save the latest model
                # into a separate file with the iteration number for external eval
                if numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving the best model...',
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                    print 'Done'

                    # save with uidx
                    if not overwrite:
                        print 'Saving the model at iteration {}...'.format(uidx),
                        saveto_uidx = '{}.iter{}.npz'.format(
                            os.path.splitext(saveto)[0], uidx)
                        numpy.savez(saveto_uidx, history_errs=history_errs,
                                    uidx=uidx, **unzip(tparams))
                        print 'Done'


                # generate some samples with the model and display them
                if numpy.mod(uidx, sampleFreq) == 0:
                    # FIXME: random selection?
                    for jj in xrange(numpy.minimum(5, x.shape[1])):
                        stochastic = True
                        sample, score = gen_sample(tparams, f_init, f_next,
                                                   x[:, jj][:, None],
                                                   model_options, trng=trng, k=1,
                                                   maxlen=30,
                                                   stochastic=stochastic,
                                                   argmax=False)
                        print 'Source ', jj, ': ',
                        for vv in x[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[0]:
                                print worddicts_r[0][vv],
                            else:
                                print 'UNK',
                        print
                        print 'Truth ', jj, ' : ',
                        for vv in y[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print
                        print 'Sample ', jj, ': ',
                        if stochastic:
                            ss = sample
                        else:
                            score = score / numpy.array([len(s) for s in sample])
                            ss = sample[score.argmin()]
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print

                # validate model on validation set and early stop if necessary
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    valid_errs = pred_probs(f_log_probs, prepare_data,
                                            model_options, valid)
                    valid_err = valid_errs.mean()
                    history_errs.append(valid_err)

                    if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                    if len(history_errs) > patience and valid_err >= \
                            numpy.array(history_errs)[:-patience].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    if numpy.isnan(valid_err):
                        ipdb.set_trace()

                    print 'Valid ', valid_err
                    fp_log.write('Valid Err = ' + str(valid_err))

                # finish after this many updates
                if uidx >= finish_after:
                    print 'Finishing after %d iterations!' % uidx
                    fp_log.write('finish after ' + str(uidx) + ' iters')
                    estop = True
                    break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err
    fp_log.write('Valid Err = ' + str(valid_err))
    fp_log.close()

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
