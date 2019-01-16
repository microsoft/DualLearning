# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from config import config_params
import os
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda%d' % (config_params.gpu)
if os.name == 'nt':
    cmdstr = '\"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe\" '
    os.system(cmdstr)
else:
    os.system(r'nvidia-smi')

from CLM.CLM import CLM_worker
from Classifier.Models import Model as Classifier
import theano
import theano.tensor as tensor
import numpy
from Util_basic import sgd_joint, prepare_data_x, unzip, itemlist_NoEmb, adadelta_joint, Optim
from Data import load_data, get_minibatches_idx, get_minibatches_idx_bucket
from collections import OrderedDict

def grad_clipping(grads, clip_c):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()
    new_grads = []
    for g in grads:
        new_grads.append(tensor.switch(g2 > (clip_c**2), g/tensor.sqrt(g2) * clip_c, g))
    return new_grads, tensor.sqrt(g2)

class monitor(object):
    def __init__(self):
        print config_params
        self.CLM = CLM_worker(lrate=1.,
                              optimizer='adadelta',
                              batch_size=config_params.minibatch,
                              saveto='model.npz',
                              validFreq=2000,
                              dispFreq=100,
                              dropout_input=config_params.CLM_drop_in,
                              dropout_output=config_params.CLM_drop_out,
                              reload_model=config_params.model_dir + '/' + config_params.model_L2S,
                              reload_option=None,
                              log='log1'
                              )
        self.classifier = Classifier(lrate=1.,  # Learning rate for sgd (not used for adadelta and rmsprop)
                                     optimizer='adadelta',
                                     saveto='model.npz',  # The best model will be saved there
                                     dispFreq=50,  # Display the training progress after this number of updates
                                     validFreq=2000,  # Compute the validation error after this number of updates
                                     batch_size=config_params.minibatch,  # The batch size during training.
                                     batch_len_threshold=None, # use dynamic batch size if sequence lengths exceed this threshold
                                     valid_batch_size=config_params.minibatch,  # The batch size used for validation/test set.
                                     lastHiddenLayer=None,
                                     dropout_output=config_params.classifier_drop_out,  
                                     dropout_input=config_params.classifier_drop_in,
                                     reload_options=None, # Path to a saved model options we want to start from
                                     reload_model=config_params.model_dir + '/' + config_params.model_S2L,
                                     embedding=None, # Path to the word embedding file (otherwise randomized)
                                     warm_LM=None,
                                     logFile='log2' # Path to log file
                                     )
        self.trainSet, self.validSet, self.testSet = \
            load_data(path=config_params.data_dir, n_words=10000, maxlen=None, sort_by_len=True, fixed_valid=True)
        self.LMscore = numpy.load(config_params.LMScoreFile)
        self.LMscore = self.LMscore[self.LMscore.files[0]]
        self.build()

    def build(self):
        LMsores = tensor.vector('LMScore', dtype='float32')
        lrate = tensor.scalar(dtype='float32')
   
        CLM_srcx, CLM_srcx_mask, CLM_ctx_, CLM_cost, CLM_sentenceLen = self.CLM.GetNll()
        classifier_x, classifier_mask, classifier_y, classifier_nlls = self.classifier.GetNll()
        consistent_loss = (((classifier_nlls + numpy.log(0.5))/CLM_sentenceLen + LMsores - CLM_cost) ** 2).mean()
        CLM_cost_avg = CLM_cost.mean()
        overall_L2S = CLM_cost_avg + config_params.trade_off_L2S * config_params.trade_off_L2S * consistent_loss
        classifier_nlls_avg = classifier_nlls.mean()
        overall_S2L = classifier_nlls_avg + config_params.trade_off_S2L * config_params.trade_off_S2L * consistent_loss

        if config_params.FreezeEmb:
            grads_L2S = tensor.grad(overall_L2S, wrt=itemlist_NoEmb(self.CLM.tparams))
        else:    
            grads_L2S = tensor.grad(overall_L2S, wrt=self.CLM.tparams.values())
        if config_params.clip_L2S > 0.:
            grads_L2S, norm_grads_L2S = grad_clipping(grads_L2S, config_params.clip_L2S)
        else:
            norm_grads_L2S = tensor.alloc(-1.)

        if config_params.FreezeEmb:
            grads_S2L = tensor.grad(overall_S2L, wrt=itemlist_NoEmb(self.classifier.tparams))
        else:    
            grads_S2L = tensor.grad(overall_S2L, wrt=self.classifier.tparams.values())
        if config_params.clip_S2L > 0.:
            grads_S2L, norm_grads_S2L = grad_clipping(grads_S2L, config_params.clip_S2L)
        else:
            norm_grads_S2L = tensor.alloc(-1.)

        if config_params.dual_style == 'all':
            merged_var_dic = OrderedDict()
            if config_params.FreezeEmb:
                merged_var_dic.update(OrderedDict((k + '_S2L',v) for (k,v) in self.classifier.tparams.iteritems() if 'Wemb' not in k ))
                merged_var_dic.update(OrderedDict((k + '_L2S',v) for (k,v) in self.CLM.tparams.iteritems() if 'Wemb' not in k ))
            else:
                merged_var_dic.update(OrderedDict((k + '_S2L',v) for (k,v) in self.classifier.tparams.iteritems()))
                merged_var_dic.update(OrderedDict((k + '_L2S',v) for (k,v) in self.CLM.tparams.iteritems()))
                
            inps = [CLM_srcx, CLM_srcx_mask, CLM_ctx_, classifier_x, classifier_mask, classifier_y, LMsores]
            outs = [CLM_cost_avg, classifier_nlls_avg, consistent_loss, overall_L2S, overall_S2L, norm_grads_L2S, norm_grads_S2L]
            self.f_grad_shared, self.f_update = Optim[config_params.optim](lrate, merged_var_dic, grads_S2L + grads_L2S, inps, outs)
        elif config_params.dual_style == 'S2L':
            if config_params.FreezeEmb:
                merged_var_dic = OrderedDict((k + '_S2L',v) for (k,v) in self.classifier.tparams.iteritems() if 'Wemb' not in k )
            else:
                merged_var_dic = OrderedDict((k + '_S2L',v) for (k,v) in self.classifier.tparams.iteritems())

            inps = [CLM_srcx, CLM_srcx_mask, CLM_ctx_, classifier_x, classifier_mask, classifier_y, LMsores]
            norm_grads_L2S = tensor.alloc(-1.)                            
            outs = [CLM_cost_avg, classifier_nlls_avg, consistent_loss, overall_L2S, overall_S2L, norm_grads_L2S, norm_grads_S2L]
            self.f_grad_shared, self.f_update = Optim[config_params.optim](lrate, merged_var_dic, grads_S2L, inps, outs)
        elif config_params.dual_style == 'L2S': 
            if config_params.FreezeEmb:
                merged_var_dic = OrderedDict((k + '_L2S',v) for (k,v) in self.CLM.tparams.iteritems() if 'Wemb' not in k )
            else:
                merged_var_dic = OrderedDict((k + '_L2S',v) for (k,v) in self.CLM.tparams.iteritems())
                
            inps = [CLM_srcx, CLM_srcx_mask, CLM_ctx_, classifier_x, classifier_mask, classifier_y, LMsores]
            norm_grads_S2L = tensor.alloc(-1.)
            outs = [CLM_cost_avg, classifier_nlls_avg, consistent_loss, overall_L2S, overall_S2L, norm_grads_L2S, norm_grads_S2L]
            self.f_grad_shared, self.f_update = Optim[config_params.optim](lrate, merged_var_dic, grads_L2S, inps, outs)
        else:
            raise Exception('Not support {} in dual_style'.format(config_params.dual_style)) 
        
    def train_one_minibatch(self, seqx, seqy, LMscore):
        CLM_x, CLM_xmask = prepare_data_x(seqx, pad_eos=True)
        labels = numpy.array(seqy).astype('int64')
        classifier_x, classifier_xmask = prepare_data_x(seqx, pad_eos=False)
        CLM_cost_avg, classifier_nlls_avg, consistent_loss, overall_L2S, overall_S2L, norm_grads_L2S, norm_grads_S2L  = self.f_grad_shared(
            CLM_x, CLM_xmask, labels, classifier_x, classifier_xmask, labels, LMscore
        )
        print 'CLM_cost_avg=%f, classifier_nlls_avg=%f, norm_grads_L2S=%f, norm_grads_S2L=%f, consistent_loss=%f,' \
              ' overall_L2S=%f, overall_S2L=%f' % (
            CLM_cost_avg, classifier_nlls_avg, norm_grads_L2S, norm_grads_S2L, consistent_loss, overall_L2S, overall_S2L )
        self.f_update(config_params.lrate)

    def train(self):
        uidx = 0
        for eidx in xrange(0, config_params.maxEpoch):
            n_samples = 0
            self.kf_train = get_minibatches_idx_bucket(self.trainSet[0],config_params.minibatch,shuffle=True)

            for _, train_index in self.kf_train:
                uidx += 1
                self.classifier.use_noise.set_value(1.)
                self.CLM.use_noise.set_value(1.)

                # Select the random examples for this minibatch
                seqx = [self.trainSet[0][t] for t in train_index]
                seqy = [self.trainSet[1][t] for t in train_index]
                LMscore = [self.LMscore[t] for t in train_index]
                self.train_one_minibatch(seqx, seqy, numpy.array(LMscore).astype('float32'))

                if uidx % config_params.validFreq == 0:
                    self.classifier.use_noise.set_value(0.)
                    self.CLM.use_noise.set_value(0.)
                    
                    if config_params.dual_style == 'all':
                        suffix_S2L = self.valid_S2L()
                        suffix_L2S = self.valid_L2S()

                        S2Lpath = config_params.model_dir + '/model_S2L_' + suffix_S2L + '_uidx' + str(uidx)
                        L2Spath = config_params.model_dir + '/model_L2S_' + suffix_L2S + '_uidx' + str(uidx)

                        numpy.savez(S2Lpath, history_errs=[], **unzip(self.classifier.tparams) )
                        numpy.savez(L2Spath, history_errs=[], **unzip(self.CLM.tparams) )
                    elif config_params.dual_style == 'S2L':
                        suffix_S2L = self.valid_S2L()
                        S2Lpath = config_params.model_dir + '/model_S2L_' + suffix_S2L + '_uidx' + str(uidx)
                        numpy.savez(S2Lpath, history_errs=[], **unzip(self.classifier.tparams) )
                    elif config_params.dual_style == 'L2S':
                        suffix_L2S = self.valid_L2S()
                        L2Spath = config_params.model_dir + '/model_L2S_' + suffix_L2S + '_uidx' + str(uidx)
                        numpy.savez(L2Spath, history_errs=[], **unzip(self.CLM.tparams) )


    def valid_S2L(self):
        acc = self.classifier.evaluate(self.trainSet, self.validSet, self.testSet)
        print 'TrainAcc=%f, ValidAcc=%f, TestAcc=%f' % (acc[0], acc[1], acc[2])
        return 'train_{}_valid_{}_test_{}'.format(acc[0], acc[1], acc[2])

    def valid_L2S(self):
        valid_ppl,  test_ppl = self.CLM.evaluate(self.validSet, self.testSet)
        print 'Valid_PPL=%f, Test_PPL=%f' % (valid_ppl, test_ppl)
        return 'valid_{}_test_{}'.format(valid_ppl, test_ppl)


if __name__ == '__main__':
    runner = monitor()
    runner.train()













