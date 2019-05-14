import tensorflow as tf
from utils import *
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib import rnn

def primary_model(x,scale):
    output = x
    
    output1 = conv2d(output,64,kernel=3,stride=(1, 1),pad=0, pad_type='zero', use_bias=True, sn=False,scope='Conv_preprocess')
    output = output1
    
    for i in range(20):
        output = rg(output,64,i)

    output = conv2d(output,64,kernel=3,stride=(1, 1),pad=0, pad_type='zero', use_bias=True, sn=False,scope='Conv_ext')
    output = tf.add(output1,output)
    output = upscale(output,scale)
    output = conv2d(output,1,kernel=3,stride=(1, 1),pad=0, pad_type='zero', use_bias=True, sn=False,scope='Conv_2')
    output = tf.nn.softsign(output)
    output += 1.0
    output /= 2.0
    return output
