# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:52:06 2018

@author: admin
"""
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

train_dir = 'data/train_for_myhat/train\hat'
train_dir2 ="data/train_for_myhat/train\person"
def get_files(file_dir,file_dir2):
    image=[]
    label=[]
    for filename in os.listdir(file_dir):
        image.append(file_dir+'/'+filename)
        label.append(0)
    for filename in os.listdir(file_dir2):
        image.append(file_dir2+'/'+filename)
        label.append(1)
    temp = np.array([image,label])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    
    return  image_list,label_list


def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    
    input_queue = tf.train.slice_input_producer([image,label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=8,capacity = capacity)
    
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
    
    


def inference(images,batch_size,n_classes):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weight',
                                  shape = [3,3,3,16],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [16],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images,weights,strides = [1,1,1,1],padding = 'SAME')
        
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name = scope.name)
        
        
        
    with tf.variable_scope('pooling_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1,ksize = [1,3,3,1],strides = [1,2,2,1],
                               padding = 'SAME',name = 'pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
                          beta = 0.75,name = 'norm1')
    
    
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weight',
                                  shape = [3,3,16,128],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [128],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1,weights,strides = [1,1,1,1],padding = 'SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name = 'conv2')
     
        
        
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
                          beta = 0.75,name = 'norm2')
        pool2 = tf.nn.max_pool(norm2,ksize = [1,3,3,1],strides = [1,1,1,1],
                               padding = 'SAME',name = 'pooling2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2,shape = [batch_size,-1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer = tf.constant_initializer(dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name = scope.name)
    
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32,
                                  initializer = tf.constant_initializer(dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(dtype=tf.float32))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name = 'local4')
    
    
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128,n_classes],
                                  dtype=tf.float32,
                                  initializer = tf.constant_initializer(dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name = 'softmax_linear')
    return softmax_linear

def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits,labels = labels,name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name = 'loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

def trainning(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy



N_CLASSES = 2
#CHANNELS=3
IMG_W = 64
IMG_H = 64
BATCH_SIZE = 16
CAPACITY = 64
MAX_STEP = 20000
learning_rate = 0.0001
#shape=[BATCH_SIZE,IMG_H,IMG_W,CHANNELS]
def run_training():
    logs_train_dir = 'picture\mylog/'
    train,train_label = get_files(train_dir,train_dir2)
    train_batch,train_label_batch = get_batch(train,train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)

    train_logits =inference(train_batch,BATCH_SIZE,N_CLASSES)

    train_loss = losses(train_logits,train_label_batch)
    train_op = trainning(train_loss,learning_rate)
    train_acc = evaluation(train_logits,train_label_batch)
    
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
            if step %  50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
                
            if step % 2000 ==0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()
            
run_training()   
    
    
    
    
    
    
    
    
    
    
