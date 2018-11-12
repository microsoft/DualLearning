Dual supervised learning for image processing.
==========

This is a tensorflow based implementation of DSL for image processing. The primal task is image classification (image-to-label) and the dual task is image generation with given label (label-to-iamge). It can leverage the dual signal of two tasks to improve the performance.

Download
----------
$ git clone https://github.com/Microsoft/DualLearning

Build
----------

**Prerequisite**

The code is build on tensorflow.


Run
----------
Traininig: refer to `example.sh`

Evaluation: refer to `batch_test_script_mainbody.sh`

Checkpoint
----------
The data and a pretrained checkpoint can be downloaded at
  https://www.dropbox.com/sh/fpnvtcmyj4mul2s/AAB4wvsxoS8pf7ExnZYe4VV1a?dl=0

Reference
----------
[1] Xia, Y., Qin, T., Chen, W., Bian, J., Yu, N., & Liu, Dual Supervised Learning, ICML2017
