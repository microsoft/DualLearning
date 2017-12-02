"""
supports simple-rnn, lstm, hierarchical lstm
supports lstm with identity skip-connections(soft), parametric skip-connections(soft)
supports resnet, resnet with identity skip-connections(full and soft), parametric skip connections(soft)
supports hybrid structure (lstm+resnet)
"""

__author__ = 'v-yirwan'

import theano.tensor as tensor
from Util import *

layers = {'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_skip': ('param_init_lstm', 'lstm_skip_layer'),
          'lstm_pskip': ('param_init_lstm_pskip', 'lstm_pskip_layer'),
          'residual': ('param_init_residual', 'residual_layer'),
          'residual_full_skip': ('param_init_residual', 'residual_full_skip_layer'),
          'residual_skip': ('param_init_residual', 'residual_skip_layer'),
          'residual_pskip': ('param_init_residual_pskip', 'residual_pskip_layer'),
          'rnn': ('param_init_rnn', 'rnn_layer'),
          'rnn_pskip': ('param_init_rnn_pskip', 'rnn_pskip_layer'),
          # modules for ResNet Modifications
          'presidual': ('param_init_presidual', 'presidual_layer'),
          'pxresidual': ('param_init_pxresidual', 'pxresidual_layer'),
          'residual_pskip_mod': ('param_init_residual_pskip', 'residual_pskip_mod_layer')
          }

def _p(pp, name):
    return '%s_%s' % (pp, name)

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# ===========================
# LSTM-related layers
# LSTM, LSTM with identity and parametric skip connections (soft)
# ===========================

def param_init_lstm(options, params, prefix='lstm', hier_level=False):
    """
    Init the LSTM parameter
    Support hierarchical architecture
    """
    if hier_level:
        # bug fixed: dimension matching for hier-mode
        W = numpy.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
    else:
        # bug fixed: different dim for embedding and hidden state
        W = numpy.concatenate([norm_weight(options['dim_word'], options['dim_proj']),
                               norm_weight(options['dim_word'], options['dim_proj']),
                               norm_weight(options['dim_word'], options['dim_proj']),
                               norm_weight(options['dim_word'], options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval[0]

def lstm_skip_layer(tparams, state_below, options, prefix='lstm_skip', mask=None):
    '''
    lstm layer with soft identity skip connections
    '''
    nsteps = state_below.shape[0]
    n_skip = options['skip_steps']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _lstm_unit(m_, x_, h_, c_, h_skip, hcnt):
        skip_flag = tensor.eq(hcnt % n_skip, 0)
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        # gates
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        # cell state
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        # new hidden stae
        h = o * tensor.tanh(c) + h_skip * skip_flag
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        # update h_skip
        h_skip = h_skip * (1-skip_flag) + h * skip_flag
        hcnt += 1

        return h, c, h_skip, hcnt

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    c = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    h_skip = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    hcnt = tensor.zeros_like(theano.shared(10.).astype('float32'))
    rval, updates = theano.scan(_lstm_unit,
                                sequences=[mask, state_below],
                                outputs_info=[h, c, h_skip, hcnt],
                                name=_p(prefix, 'layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    # return all hidden states h(t)
    return rval[0]

def param_init_lstm_pskip(options, params, prefix='lstm_pskip', hier_level=False):
    """
    Init the LSTM-pskip parameter
    """
    # same as vanilla lstm layer
    params = param_init_lstm(options, params, prefix=prefix, hier_level=hier_level)
    # weight for skip connection
    params[_p(prefix, 'W_skip')] = numpy.array([numpy.random.random_sample()]).astype('float32')[0]
    # random value in (0,1)

    return params

def lstm_pskip_layer(tparams, state_below, options, prefix='lstm_pskip', mask=None):
    '''
    lstm layer with soft parametric weighted skip connections
    '''
    nsteps = state_below.shape[0]
    n_skip = options['skip_steps']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _lstm_unit(m_, x_, h_, c_, h_skip, hcnt):
        '''
        lstm_soft_pskip unit at each time step
        :param m_: mask
        :param x_: x(t) input
        :param h_: h(t-1) recurrent hidden state
        :param c_: c(t-1) cell state
        :param h_skip: h(t-n_skip) for skip connection
        :param hcnt: mark current time stamp (to determine whether skip connection exists)
        :return: h(t), c(t), h_skip, hcnt
        '''
        skip_flag = tensor.eq(hcnt % n_skip, 0)
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        # gates
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        # cell state
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        # new hidden stae
        h = o * tensor.tanh(c) + h_skip * skip_flag * tparams[_p(prefix, 'W_skip')]
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        # update h_skip
        h_skip = h_skip * (1-skip_flag) + h * skip_flag
        hcnt += 1 # bug fixed T^T

        return h, c, h_skip, hcnt

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    c = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    h_skip = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    hcnt = tensor.zeros_like(theano.shared(10.).astype('float32'))
    rval, updates = theano.scan(_lstm_unit,
                                sequences=[mask, state_below],
                                outputs_info=[h, c, h_skip, hcnt],
                                name=_p(prefix, 'layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    # return all hidden states h(t)
    return rval[0]


# ===========================
# ResNet-related layers
# ResNet, ResNet with identity skip connections (full and soft),
# ResNet with parametric skip connections(soft)
# ===========================

def param_init_residual(options, params, prefix='residual'):
    """
    Init the residual_network parameter:
    """
    # weight for input x
    depth = options['unit_depth']
    Wx = dict()
    for idx in xrange(depth):
        Wx[idx] = norm_weight(options['dim_word'], options['dim_proj'])
    W = numpy.concatenate([ww for kk, ww in Wx.iteritems()], axis=1)
    params[_p(prefix, 'W')] = W
    b = numpy.zeros((depth * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    # weight for identity connection
    '''
    w_res = numpy.array([numpy.random.random_sample()]).astype('float32')[0]
    params[_p(prefix, 'w_res')] = w_res.astype(config.floatX)
    '''
    # weight for inter-states
    for idx in xrange(depth):
        U = ortho_weight(options['dim_proj'])
        params[_p(prefix, 'U'+str(idx+1))] = U

    return params

def residual_layer(tparams, state_below, options, prefix='residual', mask=None):
    '''
    vanilla residual layer (recurrent depth adjustable)
    '''
    # here state_below in x_emb
    nsteps = state_below.shape[0]
    depth = options['unit_depth']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _resblock(m_, x_, h_):
        y = h_
        for idx in xrange(depth):
            hy = tensor.dot(y, tparams[_p(prefix, 'U'+str(idx+1))])
            # y(i) = sigmoid(Wx(t)+b + Uy(i-1))
            y = tensor.nnet.sigmoid(_slice(x_, idx, options['dim_proj']) + hy)
        h = tensor.tanh(h_ + y)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h

    # state_below = W*x(t)+b (for all inter_state y)
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    h = tensor.alloc(numpy_floatX(0.), n_samples, options['dim_proj'])
    rval, updates = theano.scan(_resblock,
                                sequences=[mask, state_below],
                                outputs_info=[h],
                                name=_p(prefix, 'layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval # bug fixed: not rval[0], attention here

def residual_full_skip_layer(tparams, state_below, options, prefix='residual_full_skip', mask=None):
    '''
    residual layer with full skip connections (direct link without weight)
    '''
    # here state_below in x_emb
    nsteps = state_below.shape[0]
    depth = options['unit_depth']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # input mask, x(t), h(t-1), H(t-1)
    def _resblock(m_, x_, h_, H_):
        y = h_
        for idx in xrange(depth):
            hy = tensor.dot(y, tparams[_p(prefix, 'U'+str(idx+1))])
            # y(i) = sigmoid(Wx(t)+b + Uy(i-1))
            y = tensor.nnet.sigmoid(_slice(x_, idx, options['dim_proj']) + hy)
        # new hidden state
        h = tensor.tanh(h_ + y + H_[:,:,0])
        h = m_[:, None] * h + (1. - m_)[:, None] * h_ # mask
        # update skip hidden matrix
        H = tensor.zeros_like(H_)
        H = tensor.set_subtensor(H[:,:,:-1], H_[:,:,1:])
        H = tensor.set_subtensor(H[:,:,-1], h)
        return h, H

    # state_below = W*x(t)+b (for all inter_state y)
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    n_skip = options['skip_steps']
    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    H = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj, n_skip)
    rval, updates = theano.scan(_resblock,
                                sequences=[mask, state_below],
                                outputs_info=[h, H],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval[0] # return all hidden states h

def residual_skip_layer(tparams, state_below, options, prefix='residual_skip', mask=None):
    '''
    residual layer with (soft) skip connections (direct link without weight)
    '''
    # here state_below in x_emb
    nsteps = state_below.shape[0]
    depth = options['unit_depth']
    dim_proj = options['dim_proj']
    n_skip = options['skip_steps']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # input mask, x(t), h(t-1), h(skip), time_idx
    def _resblock(m_, x_, h_, h_skip, hcnt):
        y = h_
        skip_flag = theano.tensor.eq(hcnt%n_skip, 0)
        for idx in xrange(depth):
            hy = tensor.dot(y, tparams[_p(prefix, 'U'+str(idx+1))])
            # y(i) = sigmoid(Wx(t)+b + Uy(i-1))
            y = tensor.nnet.sigmoid(_slice(x_, idx, options['dim_proj']) + hy)
        # new hidden state
        h = tensor.tanh(h_ + y + h_skip*skip_flag)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_ # mask
        # update h(skip)
        h_skip = h_skip*(1-skip_flag) + h*skip_flag
        hcnt += 1
        return h, h_skip, hcnt

    # state_below = W*x(t)+b (for all inter_state y)
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    h_skip = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    # fixme: 0-dim init
    hcnt = tensor.zeros_like(theano.shared(10.).astype('float32'))
    rval, updates = theano.scan(_resblock,
                                sequences=[mask, state_below],
                                outputs_info=[h, h_skip, hcnt],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval[0] # return all hidden states h

def param_init_residual_pskip(options, params, prefix='residual_pskip'):
    """
    Init the residual network with parametric weighted skip connections:
    """
    # weight for input x
    depth = options['unit_depth']
    Wx = dict()
    for idx in xrange(depth):
        Wx[idx] = norm_weight(options['dim_word'], options['dim_proj'])
    W = numpy.concatenate([ww for kk, ww in Wx.iteritems()], axis=1)
    params[_p(prefix, 'W')] = W
    b = numpy.zeros((depth * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    # weight for skip connection
    params[_p(prefix, 'W_skip')] = numpy.array([numpy.random.random_sample()]).astype('float32')[0]
    # random value in (0,1)

    # weight for inter-states
    for idx in xrange(depth):
        U = ortho_weight(options['dim_proj'])
        params[_p(prefix, 'U'+str(idx+1))] = U
    return params

def residual_pskip_layer(tparams, state_below, options, prefix='residual_pskip', mask=None):
    '''
    residual layer with soft parametric weighted skip connections
    '''
    # here state_below in x_emb
    nsteps = state_below.shape[0]
    depth = options['unit_depth']
    dim_proj = options['dim_proj']
    n_skip = options['skip_steps']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # input mask, x(t), h(t-1), h(skip), time_idx
    def _resblock(m_, x_, h_, h_skip, hcnt):
        y = h_
        skip_flag = theano.tensor.eq(hcnt%n_skip, 0)
        for idx in xrange(depth):
            hy = tensor.dot(y, tparams[_p(prefix, 'U'+str(idx+1))])
            # y(i) = sigmoid(Wx(t)+b + Uy(i-1))
            y = tensor.nnet.sigmoid(_slice(x_, idx, options['dim_proj']) + hy)
        # new hidden state
        h = tensor.tanh(h_ + y + h_skip * skip_flag * tparams[_p(prefix, 'W_skip')])
        h = m_[:, None] * h + (1. - m_)[:, None] * h_ # mask
        # update h(skip)
        h_skip = h_skip*(1-skip_flag) + h*skip_flag
        hcnt += 1
        return h, h_skip, hcnt

    # state_below = W*x(t)+b (for all inter_state y)
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    h_skip = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    # fixme: 0-dim init
    hcnt = tensor.zeros_like(theano.shared(10.).astype('float32'))
    rval, updates = theano.scan(_resblock,
                                sequences=[mask, state_below],
                                outputs_info=[h, h_skip, hcnt],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval[0] # return all hidden states h


# ===========================
# RNN-related layers
# simple rnn and rnn with parametric skip connections (soft)
# ===========================

def param_init_rnn(options, params, prefix='rnn', hier_level=False):
    '''
    Initialize parameters for simple rnn unit
    Support hierarchical architecture
    '''
    if hier_level:
        W = ortho_weight(options['dim_proj'])
    else:
        W = norm_weight(options['dim_word'], options['dim_proj'])
    params[_p(prefix, 'W')] = W
    U = ortho_weight(options['dim_proj'])
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((options['dim_proj']))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def rnn_layer(tparams, state_below, options, prefix='rnn', mask=None):
    nsteps = state_below.shape[0]
    dim_proj = options['dim_proj']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    # input: mask, x(t), h(t-1)
    def _rnn_unit(m_, x_, h_):
        h = tensor.tanh(tensor.dot(x_, tparams[_p(prefix, 'W')]) +
                        tensor.dot(h_, tparams[_p(prefix, 'U')]) +
                        tparams[_p(prefix, 'b')])
        h = m_[:, None] * h + (1.-m_)[:, None] * h_ # mask
        return h

    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    rval, updates = theano.scan(_rnn_unit,
                                sequences=[mask, state_below],
                                outputs_info=[h],
                                name=_p(prefix, 'layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval

def param_init_rnn_pskip(options, params, prefix='rnn_pskip', hier_level=False):
    '''
    Initialize parameters for simple-rnn unit with parametric soft skip connections
    '''
    # weight for vanilla simple-rnn
    params = param_init_rnn(options, params, prefix=prefix, hier_level=hier_level)
    # weight for skip connection
    params[_p(prefix, 'W_skip')] = numpy.array([numpy.random.random_sample()]).astype('float32')[0]

    return params

def rnn_pskip_layer(tparams, state_below, options, prefix='rnn_pskip', mask=None):
    nsteps = state_below.shape[0]
    n_skip = options['skip_steps']
    dim_proj = options['dim_proj']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    assert mask is not None

    def _rnn_pskip(m_, x_, h_, h_skip, hcnt):
        skip_flag = tensor.eq(hcnt % n_skip, 0)
        h = tensor.tanh(tensor.dot(x_, tparams[_p(prefix, 'W')]) +
                        tensor.dot(h_, tparams[_p(prefix, 'U')]) +
                        tparams[_p(prefix, 'b')] +
                        skip_flag * h_skip * tparams[_p(prefix, 'W_skip')])
        h = m_[:, None] * h + (1.-m_)[:, None] * h_
        h_skip = skip_flag * h + (1-skip_flag) * h_skip
        hcnt += 1
        return h, h_skip, hcnt

    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    h_skip = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    hcnt = tensor.zeros_like(theano.shared(10.).astype('float32'))
    rval, updates = theano.scan(_rnn_pskip,
                                sequences=[mask, state_below],
                                outputs_info=[h, h_skip, hcnt],
                                name=_p(prefix, 'layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval[0]


# ===========================
# ResNet modifications
# ===========================

def residual_pskip_mod_layer(tparams, state_below, options, prefix='residual_pskip_mod', mask=None):
    '''
    residual layer with soft parametric weighted skip connections
    modifications on original pskip model
    '''
    # here state_below in x_emb
    nsteps = state_below.shape[0]
    depth = options['unit_depth']
    dim_proj = options['dim_proj']
    n_skip = options['skip_steps']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # input mask, x(t), h(t-1), h(skip), time_idx
    def _resblock_mod(m_, x_, h_, h_skip, hcnt):
        y = h_
        skip_flag = theano.tensor.eq(hcnt % n_skip, 0)
        for idx in xrange(depth):
            hy = tensor.dot(y, tparams[_p(prefix, 'U'+str(idx+1))])
            # y(i) = sigmoid(Wx(t)+b + Uy(i-1))
            y = tensor.nnet.sigmoid(_slice(x_, idx, options['dim_proj']) + hy)
        # modification: skip connection after activation
        h = tensor.tanh(h_ + y) + h_skip * skip_flag * tparams[_p(prefix, 'W_skip')]
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        h_skip = h_skip*(1-skip_flag) + h*skip_flag
        hcnt += 1
        return h, h_skip, hcnt

    # state_below = W*x(t)+b (for all inter_state y)
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    h = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    h_skip = tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)
    # fixme: 0-dim init
    hcnt = tensor.zeros_like(theano.shared(10.).astype('float32'))
    rval, updates = theano.scan(_resblock_mod,
                                sequences=[mask, state_below],
                                outputs_info=[h, h_skip, hcnt],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=options['truncate_grad'])
    return rval[0] # return all hidden states h

def param_init_presidual(options, params, prefix='presidual', nin=None, dim=None):
    """
    Init the parametric_residual_network parameter:
    """
    if nin is None:
        nin = options['dim_word']
    if dim is None:
        dim = options['dim_proj']

    # weight for input x
    depth = options['unit_depth']
    Wx = dict()
    for idx in xrange(depth):
        Wx[idx] = norm_weight(nin, dim)
    W = numpy.concatenate([ww for kk, ww in Wx.iteritems()], axis=1)
    params[_p(prefix, 'W')] = W
    b = numpy.zeros((depth * dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    w_res = rand_weight(dim, 1)
    params[_p(prefix, 'w_res')] = w_res.astype(config.floatX)
    b_res = numpy.array([numpy.random.random_sample()]).astype('float32')[0]
    params[_p(prefix, 'b_res')] = b_res

    # weight for inter-states
    for idx in xrange(depth):
        U = ortho_weight(dim)
        params[_p(prefix, 'U'+str(idx+1))] = U
    return params

def presidual_layer(tparams, state_below, options, prefix='presidual', mask=None,
                    one_step=False, init_state=None, **kwargs):
    '''
    parametric residual layer (recurrent depth adjustable)
    parametric vector on identity connection
    '''
    if one_step:
        assert init_state, 'previous state must be provided'

    # here state_below in x_emb
    nsteps = state_below.shape[0]
    depth = options['unit_depth']
    dim = options['dim_proj']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # input mask, x(t), h(t-1)
    def _presblock(m_, x_, h_):
        y = h_
        for idx in xrange(depth):
            hy = tensor.dot(y, tparams[_p(prefix, 'U'+str(idx+1))])
            y = tensor.nnet.sigmoid(_slice(x_, idx, dim) + hy)
        # p = 2*sigmoid(wh(t-1)+b)-1
        p = 2 * tensor.nnet.sigmoid(tensor.dot(h_, tparams[_p(prefix, 'w_res')]) + tparams[_p(prefix, 'b_res')]) - 1
        p_vec = p.reshape(p.shape[0], 1)
        # h(t) = tanh(ph(t-1)+y)
        h = tensor.tanh(tensor.dot(tensor.nlinalg.alloc_diag(p_vec), h_) + y)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h

    # state_below = W*x(t)+b (for all inter_state y)
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    if init_state is None:
        init_state = tensor.alloc(numpy_floatX(0.), n_samples, dim)

    if one_step:
        rval = _presblock(mask, state_below, init_state)
    else:
        rval, updates = theano.scan(_presblock,
                                    sequences=[mask, state_below],
                                    outputs_info=[init_state],
                                    name=_p(prefix, 'layers'),
                                    n_steps=nsteps)
    # rval = [rval] # note: for consistency among model layers
    return rval

def param_init_pxresidual(options, params, prefix='pxresidual', nin=None, dim=None):
    """
    Init the parametric (with respect to input) residual network parameter:
    """
    if nin is None:
        nin = options['dim_word']
    if dim is None:
        dim = options['dim_proj']

    # weight for input x
    depth = options['unit_depth']
    Wx = dict()
    for idx in xrange(depth):
        Wx[idx] = norm_weight(nin, dim)
    W = numpy.concatenate([ww for kk, ww in Wx.iteritems()], axis=1)
    params[_p(prefix, 'W')] = W
    b = numpy.zeros((depth * dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    w_res = rand_weight(dim, 1)
    params[_p(prefix, 'w_res')] = w_res.astype(config.floatX)
    u_res = rand_weight(nin, 1)
    params[_p(prefix, 'u_res')] = u_res.astype(config.floatX)
    b_res = numpy.array([numpy.random.random_sample()]).astype('float32')[0]
    params[_p(prefix, 'b_res')] = b_res

    # weight for inter-states
    for idx in xrange(depth):
        U = ortho_weight(dim)
        params[_p(prefix, 'U'+str(idx+1))] = U
    return params

def pxresidual_layer(tparams, state_below, options, prefix='pxresidual', mask=None,
                   one_step=False, init_state=None, **kwargs):
    '''
    parametric (with respect to input) residual layer (recurrent depth adjustable)
    parametric vector on identity connection
    '''
    if one_step:
        assert init_state, 'previous state must be provided'

    # here state_below in x_emb
    nsteps = state_below.shape[0]
    depth = options['unit_depth']
    dim = options['dim_proj']
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # input mask, x(t), h(t-1)
    def _presblock(m_, x_, px_, h_):
        y = h_
        for idx in xrange(depth):
            hy = tensor.dot(y, tparams[_p(prefix, 'U'+str(idx+1))])
            y = tensor.nnet.sigmoid(_slice(x_, idx, dim) + hy)
        # p = 2 * sigmoid(wh(t-1) + (ux(t)+b)) - 1
        p = 2 * tensor.nnet.sigmoid(tensor.dot(h_, tparams[_p(prefix, 'w_res')]) + px_) - 1
        p_vec = p.reshape(p.shape[0], 1)
        # h(t) = tanh(p*h(t-1) + y)
        h = tensor.tanh(tensor.dot(tensor.nlinalg.alloc_diag(p_vec), h_) + y)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h

    # state_below_x = W*x(t)+b (for all inter_state y)
    state_below_x = tensor.dot(state_below, tparams[_p(prefix, 'W')]) \
                     + tparams[_p(prefix, 'b')]
    # state_below_px = u_res*x(t)+b_res (for parametric weight on identity connection)
    state_below_px = tensor.dot(state_below, tparams[_p(prefix, 'u_res')]) \
                     + tparams[_p(prefix, 'b_res')]

    if init_state is None:
        init_state = tensor.alloc(numpy_floatX(0.), n_samples, dim)

    if one_step:
        rval = _presblock(mask, state_below_x, state_below_px, init_state)
    else:
        rval, updates = theano.scan(_presblock,
                                    sequences=[mask, state_below_x, state_below_px],
                                    outputs_info=[init_state],
                                    name=_p(prefix, 'layers'),
                                    n_steps=nsteps)
    # rval = [rval] # note: for consistency among model layers
    return rval