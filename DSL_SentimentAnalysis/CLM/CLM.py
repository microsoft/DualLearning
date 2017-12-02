from nmt_base import *
from Data import *

def _p(pp, name):
    return '%s_%s' % (pp, name)

class CLM_worker(object):
    def __init__(self,
                 round = 0,
                 dim_word=500,  # word vector dimensionality
                 dim_proj=1024,  # the number of GRU units
                 encoder='lstm',
                 patience=10,  # early stopping patience
                 max_epochs=5000,
                 finish_after=10000000000000,  # finish after this many updates
                 decay_c=-1.,  # L2 weight decay penalty
                 clip_c=5.,
                 lrate=1.,
                 n_words=10000,    # vocabulary size
                 maxlen=None,  # maximum length of the description
                 minlen=1,
                 start_iter=0,
                 start_epoch=0,
                 optimizer='adadelta',
                 batch_size=16,
                 valid_batch_size=16,
                 saveto='model.npz',
                 validFreq=2000,
                 dispFreq=100,
                 saveFreq=100000,  # save the parameters after every saveFreq updates
                 newDumpFreq=10000,
                 syncFreq = 500000000000,
                 sampleFreq=10000000000,  # generate some samples after every sampleFreq
                 valid_dataset=None,
                 test_dataset=None,
                 dictionary=None,
                 sampleFileName="sampleFile.txt",
                 embedding=None,
                 dropout_input=None,
                 dropout_output=None,
                 reload_model=None,
                 reload_option=None,
                 log=None,
                 monitor_grad=False,
                 pad_sos=False):
            # Model options
        if pad_sos:
            n_words += 1
        self.options = locals().copy()

        print('log = ', log)
        F_log = open(log, "a")

        voc_size = n_words - 1 if pad_sos else n_words

        # reload options
        if reload_option is not None and os.path.exists(reload_option):
            print "Reloading model options...",
            with open('%s' % reload_option, 'rb') as f:
                model_options = pkl.load(f)
            print "Done"

        # init parameters
        print 'Initializing model parameters...',
        params = init_lm_params(self.options)
        print 'Done'

        # load pre-trained word embedding
        if embedding is not None and os.path.exists(embedding):
            print 'Load Embedding from ', embedding
            Wemb = numpy.array(numpy.load(open(embedding, "rb")))
            assert Wemb.shape[0] == self.options['n_words']
            assert Wemb.shape[1] == self.options['dim_word']
            print 'Using pre-trained word embedding...',
            params['Wemb'] = Wemb.astype(numpy.float32)
            print 'vocab size', params['Wemb'].shape[0], ', dim', params['Wemb'].shape[1]

        # reload parameters
        if reload_model is not None and os.path.exists(reload_model):
            print "Reloading model parameters...",
            params = load_params(reload_model, params)
            print "Done"

        # create shared variables for parameters
        self.tparams = init_tparams(params)

        # build the symbolic computational graph
        print 'Building model...'
        self.trng = RandomStreams(1234)
        self.use_noise = theano.shared(numpy.float32(0.))

    def GetNll(self):
        srcx, srcx_mask, ctx_, cost, sentenceLen = self.build_lm_model()
        print 'Done'

        print 'Building f_log_probs',
        self.f_log_probs = theano.function([srcx, srcx_mask, ctx_], cost, profile = profile)
        print 'Done'
        return srcx, srcx_mask, ctx_, cost, sentenceLen

    # build a training model
    def build_lm_model(self):
        srcx = tensor.matrix('x', dtype='int64')
        srcx_mask = tensor.matrix('x_mask', dtype='float32')
        ctx_ = tensor.vector('ctx_', dtype='int64')
        x = srcx[:-1, :]
        y = srcx[1:,:]

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]
        print('check init ok')
        emb  = self.tparams['Wemb'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.options['dim_word']])
        emb_ctx = self.tparams['Wemb_ctx'][ctx_].reshape([n_samples, self.options['dim_word']])
        print('check embed ok')
        # input

        if self.options['dropout_input'] is not None and self.options['dropout_input'] > 0:
            print 'Applying drop-out on input embedding (dropout_input:', self.options['dropout_input'], ")"
            emb = dropout_layer(emb, self.use_noise, self.trng, self.options['dropout_input'])
            emb_ctx = dropout_layer(emb_ctx, self.use_noise, self.trng, self.options['dropout_input'])

        init_state = tensor.alloc(0., n_samples, self.options['dim_proj'])
        init_cell  = tensor.alloc(0., n_samples, self.options['dim_proj'])

        # pass through gru layer, recurrence here
        print 'Using', self.options['encoder'], 'unit for encoder'
        print 'Training with successive sentences'
        init_states = [init_state, init_cell]
        proj = lstm_layer(self.tparams, emb, emb_ctx, self.options,
                          prefix='encoder',
                          init_state=init_state,
                          cell_state=init_cell,
                          mask = srcx_mask[:-1,:])


        proj_h = proj[0] # all hidden states

        next_states = [st[-1] for st in proj] # first last hidden_state, second last cell_state

        if self.options['dropout_output'] is not None and self.options['dropout_output'] > 0:
            print 'Applying drop-out on hidden states (dropout_proj:', self.options['dropout_output'], ")"
            proj_h = dropout_layer(proj_h, self.use_noise, self.trng, self.options['dropout_output'])


        # compute word probabilities
        def _prob(proj_h):
            logit_lstm = get_layer('ff')[1](self.tparams, proj_h, self.options, prefix='ff_logit_lstm', activ='linear')
            logit_prev = get_layer('ff')[1](self.tparams, emb, self.options, prefix='ff_logit_prev', activ='linear')
            logit_label = get_layer('ff')[1](self.tparams, emb_ctx, self.options, prefix='ff_logit_label', activ='linear')
            logit = tensor.tanh(logit_lstm + logit_prev + logit_label)

            #logit = tensor.tanh(logit_lstm)
            # split to calculate
            logit = get_layer('ff')[1](self.tparams, logit, self.options, prefix='ff_logit', activ='linear')
            logit_shp = logit.shape # n_timesteps * n_samples * n_words
            probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))
            return probs

        probs = _prob(proj_h)

        # cost
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.options['n_words'] + y_flat

        # probs:(seq,batch,worddim) <-> x:(seq,batch) become the right place value
        # y:(seq_len, batch_size)
        def _cost(probs):
            cost = -tensor.log(probs.flatten()[y_flat_idx] + 1e-10)
            cost = cost.reshape([y.shape[0], y.shape[1]])
            sentenceLen = srcx_mask[1:,:].sum(axis=0)
            cost = (cost * srcx_mask[1:, :]).sum(axis=0) / sentenceLen
            return cost, sentenceLen

        cost, sentenceLen = _cost(probs)

        return srcx, srcx_mask, ctx_, cost, sentenceLen #(seq, batch, worddim)

    # calculate the log probablities on a given corpus using language model
    def pred_probs(self, valid_Data, valid_batch_size):
        self.use_noise.set_value(0.)
        nlls = []
        dataLen = []
        valid_x, valid_y = valid_Data[0], valid_Data[1]

        for idx in xrange((len(valid_x) + valid_batch_size - 1) // valid_batch_size ):
            data = valid_x[idx * valid_batch_size : (idx + 1) * valid_batch_size]
            label = valid_y[idx * valid_batch_size : (idx + 1) * valid_batch_size]
            dataLen += [len(tt) for tt in data]
            x, x_mask = prepare_data_x(data, pad_sos=self.options['pad_sos'], n_word=self.options['n_words'])
            cost = self.f_log_probs(x, x_mask, numpy.array(label).astype('int64'))
            nlls += cost.tolist()

        nlls = numpy.array(nlls).astype('float32')
        dataLen = numpy.array(dataLen).astype('float32')
        return numpy.exp((nlls * dataLen).sum() / dataLen.sum())

    def evaluate(self, validSet, testSet):
        valid_ppl = self.pred_probs(validSet, 32)
        test_ppl = self.pred_probs(testSet, 32)
        return valid_ppl,  test_ppl


'''
def train(round = 0,
        dim_word=1000,  # word vector dimensionality
        dim_proj=1000,  # the number of GRU units
        encoder='lstm',
        patience=10,  # early stopping patience
        max_epochs=5000,
        finish_after=10000000000000,  # finish after this many updates
        decay_c=0.,  # L2 weight decay penalty
        clip_c=5.,
        lrate=1.,
        n_words = 10000,    # vocabulary size
        maxlen=None,  # maximum length of the description
        minlen=1,
        start_iter=0,
        start_epoch=0,
        optimizer='adadelta',
        batch_size=32,
        valid_batch_size=20,
        saveto='model.npz',
        validFreq=1000,
        dispFreq=100,
        saveFreq=1000,  # save the parameters after every saveFreq updates
        newDumpFreq=10000,
        syncFreq = 50,
        sampleFreq=100,  # generate some samples after every sampleFreq
        sampleNum = 50, # generate sampleNum sentences
        dataset=None,
        valid_dataset=None,
        test_dataset=None,
        dictionary=None,
        sampleFileName="sampleFile.txt",
        embedding=None,
        dropout_input=None,
        dropout_output=None,
        reload_model=None,
        reload_option=None,
        log=None,
        monitor_grad=False,
        pad_sos=False):

    # Model options
    if pad_sos:
        n_words += 1
    model_options = locals().copy()
    print "model options:"
    for kk, vv in model_options.iteritems():
        print "\t"+kk+":\t"+str(vv)

    print('log = ', log)
    F_log = open(log, "a")

    if start_iter == 0:
        F_log.write("model options:\n")
        for kk, vv in model_options.iteritems():
            F_log.write("\t"+kk+":\t"+str(vv)+"\n")
        F_log.write("-----------------------------------------\n\n")


    print 'Loading training dataset...'

    voc_size = n_words - 1 if pad_sos else n_words

    trainSet, validSet, testSet = load_data(path=dataset, n_words=n_words, maxlen=maxlen, sort_by_len=True, fixed_valid=True)

    # reload options
    if reload_option is not None and os.path.exists(reload_option):
        print "Reloading model options...",
        with open('%s' % reload_option, 'rb') as f:
            model_options = pkl.load(f)
        print "Done"

    # init parameters
    print 'Initializing model parameters...',
    params = init_lm_params(model_options)
    print 'Done'

    # load pre-trained word embedding
    if embedding is not None and os.path.exists(embedding):
        print 'Load Embedding from ', embedding
        Wemb = numpy.array(numpy.load(open(embedding, "rb")))
        if Wemb.shape[0] == model_options['n_words'] and Wemb.shape[1] == model_options['dim_word']:
            print 'Using pre-trained word embedding...',
            params['Wemb'] = Wemb.astype(numpy.float32)
            print 'vocab size', params['Wemb'].shape[0], ', dim', params['Wemb'].shape[1]

    # reload parameters
    if reload_model is not None and os.path.exists(reload_model):
        print "Reloading model parameters...",
        params = load_params(reload_model, params)
        print "Done"

    # create shared variables for parameters
    tparams = init_tparams(params)

    # build the symbolic computational graph
    print 'Building model...'
    trng, use_noise, srcx, srcx_mask, ctx_, cost = build_lm_model(tparams, model_options)

    print 'Building f_log_probs',
    f_log_probs = theano.function([srcx, srcx_mask, ctx_], cost, profile = profile)
    print 'Done'
    cost = cost.mean(axis=0)
    # apply L2 regularization on weights
    if decay_c > 0.:
        print "Applying L2 regularization (decay_c: "+str(decay_c)+')...',
        cost = l2_regularization(tparams, cost, decay_c)
        print "Done"

    # after any regularizer - compile the computational graph for cost
    print 'Building f_cost',
    f_cost = theano.function([srcx, srcx_mask, ctx_], cost, profile = profile)
    print 'Done'

    print 'Computing gradient',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        print 'Applying gradient clipping (clip_c:'+str(clip_c)+')...',
        grads = grad_clipping(grads, clip_c)
        print 'Done'

    # compile the optimizer, the actual computational graph is compiled here
    print 'Building optimizers...',
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, [srcx, srcx_mask, ctx_], cost)
    #f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    sys.stdout.flush()

    history_errs = []
    # reload history
    if reload_model is not None and os.path.exists(reload_model):
        history_errs = list(numpy.load(reload_model)['history_errs'])
    best_p = None
    bad_count = 0

    # Training loop
    bad_counter = 0
    uidx = start_iter
    estop = False
    start_time = time.time()
    n_samples = 0
    cost_accu = 0

    for eidx in xrange(start_epoch, max_epochs):
        epoch_start_time = time.time()
        print "Start epoch ", eidx
        n_samples = 0

        print 'Start epoch', eidx
        epoch_start_time = time.time()
        n_samples = 0

        kf_train = get_minibatches_idx(len(trainSet[0]), batch_size, shuffle=True)

        for _, train_index in kf_train:
            uidx += 1
            x = [trainSet[0][t] for t in train_index]
            y = [trainSet[1][t] for t in train_index]
            n_samples += len(x)
            use_noise.set_value(1.) #training mode

            # pad batch and create mask
            x, x_mask = prepare_data_x(x, pad_eos=True,pad_sos=model_options['pad_sos'],n_word=model_options['n_words'])

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y) # input argument issue fixed

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                F_log.write("=========================================\nNaN detected\n")
                F_log.write('Epoch'+str(eidx)+'\tIter '+str(uidx)+'\tBatch Length '+str(x.shape[0])+'\n')
                return 1.

            cost_accu += cost
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, '\tIter ', uidx, '\tLoss ', cost_accu/float(dispFreq), '\tUD ', ud,
                print '\tLength', x.shape[0], '\tSize ', x.shape[1]
                F_log.write('Epoch '+str(eidx)+'\tIter '+str(uidx)+'\tLoss '+str(cost_accu/float(dispFreq))
                        +'\tUD '+str(ud)+'\tLength '+str(x.shape[0])+'\tSize '+str(x.shape[1])+'\n')
                cost_accu = 0
                sys.stdout.flush()

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                print "Validating...",
                use_noise.set_value(0.)
                # fixed for successive mode
                valid_ppl = pred_probs(f_log_probs, prepare_data_x, model_options, validSet, batch_size)
                history_errs.append(valid_ppl)
                print "Done"

                if uidx == 0 or valid_ppl <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_ppl >= numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        F_log.write('##############\nEarly Stop!\n##############\n')
                        estop = True
                        break

                # perplexity

                test_ppl = pred_probs(f_log_probs, prepare_data_x, model_options, testSet, batch_size)

                print 'Perplexity: { Valid', valid_ppl, ', Test', test_ppl, '}'
                F_log.write('Perplexity: Valid '+str(valid_ppl)+'\tTest '+str(test_ppl)+'\n')
                F_log.write('====================================\n')
                sys.stdout.flush()

                # save the current models
                savefile = saveto + "_e" + str(eidx) + "_i" + str(uidx) + "_valid_" + str(valid_ppl) + '_test_' + str(test_ppl)
                numpy.savez(savefile, history_errs=history_errs, **unzip(tparams))
                pkl.dump(model_options, open('%s.option.pkl' % saveto, 'wb'))

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                F_log.write('##############\nFinishing after '+str(uidx)+' iterations!\n##############\n')
                estop = True
                break

        epoch_end_time = time.time()
        print 'Epoch', eidx, 'completed, Seen', n_samples, 'samples, Time', epoch_end_time-epoch_start_time
        F_log.write("-----------------------------------------------------------\n")
        F_log.write("Epoch "+str(eidx)+" completed, Seen "+str(n_samples)+" samples, Time "+str(epoch_end_time-epoch_start_time)+"\n")
        F_log.write("------------------------------------------------------------\n")

        if estop:
            break

    end_time = time.time()
'''