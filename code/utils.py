import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

wt_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
wt_reg = None


def batch_norm(X,is_training,i):
    output = tf.contrib.layers.batch_norm(X,
                              updates_collections=None,
                              decay=0.9,
                              center=True,
                              scale=True,
                              is_training=is_training,
                              trainable=is_training,
                              scope='BN_'+str(i),
                              reuse=tf.AUTO_REUSE,
                              fused=True,
                              zero_debias_moving_mean=True,
                              adjustment = lambda shape: ( tf.random_uniform(shape[-1:], 0.93, 1.07), tf.random_uniform(shape[-1:], -0.1, 0.1)),
                              renorm=False)
    return output

def c_attention(x,ch,i,j,r):
    out = tf.keras.layers.GlobalMaxPool2D()(x)
    out = tf.expand_dims(out,axis=1)
    out = tf.expand_dims(out,axis=1)
    out = conv2d(out, ch, kernel=1, stride=(1,1), pad=0, pad_type='zero', use_bias=True, sn=False, scope='ca_conv1_'+str(i)+'_'+str(j))
    out = conv2d(out, int(ch/r),stride=(1,1), pad=0, pad_type='zero', use_bias=True, sn=False,scope='ca_conv5_'+str(i)+'_'+str(j))
    out = tf.nn.relu(out)
    out = conv2d(out, ch, kernel=1, stride=(1,1), pad=0, pad_type='zero', use_bias=True, sn=False, scope='ca_conv2_'+str(i)+'_'+str(j))
    out = conv2d(out, ch,kernel=1, stride=(1,1),pad=0, pad_type='zero', use_bias=True, sn=False,scope='ca_conv6_'+str(i)+'_'+str(j))
    out = tf.nn.sigmoid(out)
    out = tf.tile(out,[1,tf.shape(x)[1],tf.shape(x)[2],1])
    out = out*x
    return out

def rcab(x,ch,i,j,r):
    # Residual Channel Attention Block
    out = conv2d(x, ch,kernel=3, stride=(1,1),pad=0, pad_type='zero', use_bias=True, sn=False,scope='rcab_conv1_'+str(i)+'_'+str(j))
    out = tf.nn.relu(out)
    out =conv2d(out, ch, kernel=3, stride=(1,1),pad=0, pad_type='zero', use_bias=True, sn=False,scope='rcab_conv2_'+str(i)+'_'+str(j))
    out = c_attention(out,ch,i,j,r)
    out+=x
    return out

def rg(x,ch,i,b=20,r=16):
    out = x
    for j in range(b):
            out = rcab(out,ch,i,j,r)
    out = conv2d(out,ch,kernel=3,stride=(1,1),pad=0, pad_type='zero', use_bias=True, sn=False,scope='rg_conv_'+str(i))
    out+=x
    return out

def conv2d(x, channels, kernel=4, stride=(1,1), pad=0,padding='VALID', pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=wt_init,
                                regularizer=wt_reg)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride[0], stride[1], 1], padding=padding)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=wt_init,
                                 kernel_regularizer=wt_reg,
                                 strides=stride, use_bias=use_bias)
        return x

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def deconv2d(x, channels, kernel=4, stride=(1,1), padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], channels]

        else:
            output_shape =[tf.shape(x)[0], x_shape[1] * stride[0] + max(kernel - stride[0], 0), x_shape[2] * stride[1] + max(kernel - stride[1], 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=wt_init, regularizer=wt_reg)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride[0], stride[1], 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=wt_init, kernel_regularizer=wt_reg,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def data_augment(image_rgb, image_y):
    opt = tf.random_uniform((),minval=0,maxval=4,dtype=tf.int32)
    image_rgb = tf.cond(tf.equal(opt,0), lambda: tf.image.rot90(image_rgb,k=1), lambda: image_rgb)
    image_rgb = tf.cond(tf.equal(opt,1), lambda: tf.image.rot90(image_rgb,k=2), lambda: image_rgb)
    image_rgb = tf.cond(tf.equal(opt,2), lambda: tf.image.rot90(image_rgb,k=3), lambda: image_rgb)
    image_y = tf.cond(tf.equal(opt,0), lambda: tf.image.rot90(image_y,k=1), lambda: image_y)
    image_y = tf.cond(tf.equal(opt,1), lambda: tf.image.rot90(image_y,k=2), lambda: image_y)
    image_y = tf.cond(tf.equal(opt,2), lambda: tf.image.rot90(image_y,k=3), lambda: image_y)
    opt = tf.random_uniform((),minval=0,maxval=2,dtype=tf.int32)
    image_rgb = tf.cond(tf.equal(opt,0), lambda: tf.image.flip_left_right(image_rgb), lambda: image_rgb)
    image_y = tf.cond(tf.equal(opt,0), lambda: tf.image.flip_left_right(image_y), lambda: image_y)
    return image_rgb, image_y
        
def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, int(c/r), r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
    X = tf.split(X, b,1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, int(c/(r**2))))

def upscale(X, r=4, color=False):
    if color:
        Xc = tf.split(X, 3,3 )
        X = tf.concat([_phase_shift(x, r) for x in Xc],3)
    else:
        X = _phase_shift(X, r)
    return X

