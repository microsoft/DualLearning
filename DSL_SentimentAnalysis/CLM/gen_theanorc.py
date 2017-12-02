import sys
import codecs

if len(sys.argv) < 3:
    raise Exception('Not enough argv')

theano_rc = r"""
[global]
mode = FAST_RUN
device = gpu
floatX = float32
on_unused_input = warn
optimizer = fast_run
#allow_gc=False
cuda.disable_gcc_cudnn_check=True

[lib]
cnmem = 0.75

[nvcc]
flags=-L{0}\libs
root=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5
fast_math = True

"""

theano_rc = theano_rc.format(sys.argv[1])

print(theano_rc)

with codecs.open(sys.argv[2], 'w', 'utf-8') as f:
    f.write(theano_rc)
