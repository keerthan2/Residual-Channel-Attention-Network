import tensorflow as tf
#from model_func import *
from model_func import *

def model_fn(features, labels, mode, params):
    with tf.variable_scope('base_model', reuse=tf.AUTO_REUSE):
        image_rgb = features['image_rgb']
        image_y = features['image_y']
        lrate = params['lrate']
        batch_size = params['batch_size'] 
        model_dir =  params['logdir']  
        scale = params['scale']

        with tf.variable_scope("main_model",initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02)):
            predicted_y = primary_model(image_rgb,scale)
            l1_loss = tf.norm(predicted_y - image_y, ord=1) 
                
        vars1 = tf.trainable_variables()
        g_params = [v for v in vars1 if v.name.startswith('main_model/')]
    
        total_loss = l1_loss 
        global_step = tf.train.get_or_create_global_step()   
        optimizer = tf.contrib.estimator.TowerOptimizer(tf.train.AdamOptimizer(lrate,beta1=0.9, beta2=0.999))
        update_step = optimizer.minimize(l1_loss,var_list=g_params,name='l1_loss_minimize',global_step=global_step)

        predicted_y = ((predicted_y)) * 255.0
        predicted_y = tf.cast(predicted_y,tf.uint8)
        image_y = ((image_y)) * 255.0
        image_y = tf.cast(image_y,tf.uint8)
        psnr = tf.reduce_mean(tf.image.psnr(image_y,predicted_y,255,name='PSNR'))
        ssim = tf.reduce_mean(tf.image.ssim(image_y,predicted_y,255))
        
        tf.summary.scalar('losses/l1_loss', l1_loss)
        tf.summary.scalar('accuracy/PSNR', psnr)
        tf.summary.scalar('accuracy/SSIM', ssim)
        tf.summary.scalar('parameter/L_Rate', lrate)
        tf.summary.image('GT/Images', image_y)
        tf.summary.image('Predicted/Images', predicted_y)
        tf.summary.image('Input/Images', image_rgb)
        
        return tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,train_op=update_step)