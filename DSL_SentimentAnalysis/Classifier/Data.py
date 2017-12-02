"""
data loading and minibatch generation
"""
__author__ = 'v-yirwan'

import cPickle as pkl
import gzip
import os
import numpy
from theano import config

def get_dataset_file(dataset, default_dataset, origin):
    '''
    Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present
    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    return dataset

def load_data(path="imdb.pkl", n_words=100000, maxlen=None,
              sort_by_len=True, fixed_valid=True, valid_portion=0.1):
    '''
    Loads the dataset
    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    :type fixed_valid: bool
    :param fixed_valid: load fixed validation set from the corpus file,
        which would otherwise be picked randomly from the training set with
        proportion [valid_portion]
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.

    '''

    # Load the dataset
    path = get_dataset_file(
        path, "imdb.pkl",
        "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = pkl.load(f)
    if fixed_valid:
        valid_set = pkl.load(f)
    test_set = pkl.load(f)
    f.close()

    def _truncate_data(train_set):
        '''
        truncate sequences with lengths exceed max-len threshold
        :param train_set: a list of sequences list and corresponding labels list
        :return: truncated train_set
        '''
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y
        return train_set

    def _set_valid(train_set, valid_portion):
        '''
        set validation with [valid_portion] proportion of training set
        '''
        train_set_x, train_set_y = train_set
        n_samples = len(train_set_x)
        sidx = numpy.random.permutation(n_samples) # shuffle data
        n_train = int(numpy.round(n_samples * (1. - valid_portion)))
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]
        train_set = (train_set_x, train_set_y)
        valid_set = (valid_set_x, valid_set_y)
        del train_set_x, train_set_y, valid_set_x, valid_set_y
        return train_set, valid_set

    if maxlen:
        train_set = _truncate_data(train_set)
        if fixed_valid:
            print 'Loading with fixed validation set...',
            valid_set = _truncate_data(valid_set)
        else:
            print 'Setting validation set with proportion:', valid_portion, '...',
            train_set, valid_set = _set_valid(train_set, valid_portion)
        test_set = _truncate_data(test_set)

    if maxlen is None and not fixed_valid:
        train_set, valid_set = _set_valid(train_set, valid_portion)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    # remove unk from dataset
    train_set_x = remove_unk(train_set_x) # use 1 if unk
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        # ranked from shortest to longest
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test

def load_mnist(path='mnist.pkl', fixed_permute=True, rand_permute=False):
    f = open(path, 'rb')
    train = pkl.load(f)
    valid = pkl.load(f)
    test = pkl.load(f)
    f.close()

    def _permute(data, perm):
        x, y = data
        x_new = []
        for xx in x:
            xx_new = [xx[pp] for pp in perm]
            x_new.append(xx_new)
        return (x_new, y)

    def _trans2list(data):
        x, y = data
        x = [list(xx) for xx in x]
        return (x, y)

    if rand_permute:
        print 'Using a fixed random permutation of pixels...',
        perm = numpy.random.permutation(range(784))
        train = _permute(train, perm)
        valid = _permute(valid, perm)
        test = _permute(test, perm)
    elif fixed_permute:
        print 'Using permuted dataset...',

    _trans2list(train)
    _trans2list(valid)
    _trans2list(test)

    return train, valid, test

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def get_minibatches_idx_bucket(dataset, minibatch_size, shuffle=False):
    """
    divide into different buckets according to sequence lengths
    dynamic batch size
    """
    # divide into buckets
    slen = [len(ss) for ss in dataset]
    bucket1000 = [sidx for sidx in xrange(len(dataset))
                  if slen[sidx] > 0 and slen[sidx] <= 1000]
    bucket3000 = [sidx for sidx in xrange(len(dataset))
                  if slen[sidx] > 1000 and slen[sidx] <= 3000]
    bucket_long = [sidx for sidx in xrange(len(dataset))
                   if slen[sidx] > 3000]

    # shuffle each bucket
    if shuffle:
        numpy.random.shuffle(bucket1000)
        numpy.random.shuffle(bucket3000)
        numpy.random.shuffle(bucket_long)

    # make minibatches
    def _make_batch(minibatches, bucket, minibatch_size):
        minibatch_start = 0
        n = len(bucket)
        for i in range(n // minibatch_size):
            minibatches.append(bucket[minibatch_start : minibatch_start + minibatch_size])
            minibatch_start += minibatch_size
        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(bucket[minibatch_start:])
        return minibatches

    minibatches = []
    _make_batch(minibatches, bucket1000, minibatch_size=minibatch_size)
    _make_batch(minibatches, bucket3000, minibatch_size=minibatch_size//2)
    _make_batch(minibatches, bucket_long, minibatch_size=minibatch_size//8)

    # shuffle minibatches
    numpy.random.shuffle(minibatches)

    return zip(range(len(minibatches)), minibatches)

def prepare_data(seqs, labels, maxlen=None, dataset='text'):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    if dataset == 'mnist':
        x = numpy.zeros((maxlen, n_samples)).astype('float32')
    else:
        x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels

def prepare_data_hier(seqs, labels, hier_len, maxlen=None, dataset='text'):
    '''
    prepare minibatch for hierarchical model
    '''
    # sort (long->short)
    sorted_idx = sorted(range(len(seqs)), key=lambda x: len(seqs[x]), reverse=True)
    seqs = [seqs[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    # truncate data
    lengths = [len(s) for s in seqs]
    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l  <maxlen :
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs
        if len(lengths) < 1:
            return None, None, None

    # set batch size
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    if maxlen % hier_len == 0:
        n_batch = maxlen/hier_len
    else:
        n_batch = maxlen//hier_len + 1
        maxlen = n_batch * hier_len

    # padding whole batch
    if dataset == 'mnist':
        x = numpy.zeros((maxlen, n_samples)).astype('float32')
    else:
        x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1

    # slice to mini-batches
    x_batch = [x[bidx*hier_len:(bidx+1)*hier_len, :] for bidx in range(n_batch)]
    if dataset == 'mnist':
        x_batch = numpy.array(x_batch).astype('float32')
    else:
        x_batch = numpy.array(x_batch).astype('int64')
    mask_batch = [x_mask[bidx*hier_len:(bidx+1)*hier_len, :] for bidx in range(n_batch)]
    mask_batch = numpy.array(mask_batch).astype(config.floatX)

    # mask for hier-level
    mask_hier = numpy.ones((n_batch, n_samples)).astype(config.floatX)
    for idx in range(n_samples):
        mpos = numpy.where(x_mask[:, idx]==0)[0]
        if len(mpos) == 0:
            continue
        bidx = min(mpos[0]//hier_len+1, n_batch)
        if mpos[0] % hier_len == 0:
            bidx -= 1 # bug fixed TODO: more elegant solution?
        mask_hier[bidx:, idx] = 0

    return x_batch, mask_batch, mask_hier, labels