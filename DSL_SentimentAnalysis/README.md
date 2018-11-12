Dual supervised learning for sentiment analysis.
==========

This is a theano-implementation of DSL for sentiment analysis. The primal task is sentiment classification (text-to-label) and the dual task is sentence generation with given label (label-to-text). It can leverage the dual signal of two tasks to improve the performance.

Download
----------
$ git clone https://github.com/Microsoft/DualLearning

Build
----------

**Prerequisite**

The code is build on python 2.7 and theano.


Run
----------
Windows: Please refer to "train.bat" and "valid.bat"
Linux:   Please refer to ou can refer to "train_linux.sh" and "valid_linux.sh"

Checkpoint
----------
A pretrained checkpoint can be downloaded at
  https://www.dropbox.com/sh/yxkaosxqxmhxnib/AAAK19n7y6cRh3WZHRNGqlXga?dl=0

Reference
----------
[1] Xia, Y., Qin, T., Chen, W., Bian, J., Yu, N., & Liu, T. Y. (2017). Dual supervised learning. ICML
