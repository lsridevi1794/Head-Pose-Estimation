from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as conMatrix
from sklearn.metrics import classification_report as ClassR
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import zoom
import os.path
import imp #to check for missing modules
import math

from six.moves import cPickle as pickle



def cnn_model(image):
    mu = 0
    sigma = 0.1
    X = tf.reshape(image, shape=[-1, 64, 64, 1])
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    F_W1 = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 64), mean = mu, stddev = sigma))
    F_b1 = tf.Variable(tf.zeros(64))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(X, F_W1, strides = [1, 1, 1, 1], padding = 'SAME'),F_b1)
    print(conv1.shape)
    
    # TODO: Activation.
    conv1 = tf.tanh(conv1)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print(norm1.shape)
    
    # TODO: Layer 2: Convolutional. Input = 32x32x1. Output = 28x28x6.
    F_W2 = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    F_b2 = tf.Variable(tf.zeros(128))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(norm1, F_W2, strides = [1, 1, 1, 1], padding = 'SAME'),F_b2)
    print(conv2.shape)
    
    # TODO: Activation.
    conv2 = tf.tanh(conv2)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print(norm2.shape)
    
    # TODO: Layer 3: Convolutional. Input = 32x32x1. Output = 28x28x6.
    F_W3 = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256), mean = mu, stddev = sigma))
    F_b3 = tf.Variable(tf.zeros(256))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(norm2, F_W3, strides = [1, 1, 1, 1], padding = 'SAME'),F_b3)
    print(conv3.shape)
    
    # TODO: Activation.
    conv3 = tf.tanh(conv3)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print(norm3.shape)
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flatten_out = flatten(norm3)
    print(flatten_out.shape)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    F_W4 = tf.Variable(tf.truncated_normal(shape=(8*8*256,256), mean = mu, stddev = sigma))
    F_b4 = tf.Variable(tf.zeros(256))
    FC1 = tf.matmul(flatten_out,F_W4)+F_b4
    print(FC1.shape)
    
    # TODO: Activation.
    FC1 = tf.tanh(FC1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    F_W5 = tf.Variable(tf.truncated_normal(shape=(256,3), mean = mu, stddev = sigma))
    F_b5 = tf.Variable(tf.zeros(3))
    FC2 = tf.matmul(FC1,F_W5)+F_b5
    print(FC2.shape)
    
    # TODO: Activation.
    # output = tf.sigmoid(FC2)
    # output = tf.nn.softmax(FC2)
    output = FC2
    return output
