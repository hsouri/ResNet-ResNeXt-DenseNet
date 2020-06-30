"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Hossein Souri (hsouri@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True


def SimpleModel(Img, ImageSize, MiniBatchSize, num_classes = 10):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the xwork
    prSoftMax - softmax output of the xwork
    """

    #############################
    # Fill your xwork here!
    #############################


    x = tf.layers.conv2d(inputs=Img, name='conv_1', padding='same', filters=16, kernel_size=5, activation=None)
    x = tf.nn.relu(x, name='Relu_1')

    x = tf.layers.conv2d(inputs=x, name='conv_2', padding='same', filters=32, kernel_size=3, activation=None)
    x = tf.nn.relu(x, name='Relu_2')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    x = tf.layers.conv2d(inputs=x, name='conv_3', padding='same', filters=64, kernel_size=3, activation=None)
    x = tf.nn.relu(x, name='Relu_3')


    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    x = tf.layers.flatten(x)

    x = tf.layers.dense(inputs=x, name='fc1', units=256, activation=tf.nn.relu)


    x = tf.layers.dense(inputs=x, name='fc3', units=num_classes, activation=None)


    return x, tf.nn.softmax(logits=x)


def ImprovedModel(Img, ImageSize, MiniBatchSize, num_classes = 10):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the xwork
    prSoftMax - softmax output of the xwork
    """

    #############################
    # Fill your xwork here!
    #############################



    x = tf.layers.conv2d(inputs=Img, name='conv_1', padding='same', filters=16, kernel_size=5, activation=None)
    x = tf.layers.batch_normalization(inputs=x, axis=-1, center=True, scale=True, name='batch_norm_1')
    x = tf.nn.relu(x, name='Relu_1')

    x = tf.layers.conv2d(inputs=x, name='conv_2', padding='same', filters=32, kernel_size=3, activation=None)
    x = tf.layers.batch_normalization(inputs=x, axis=-1, center=True, scale=True, name='batch_norm_2')
    x = tf.nn.relu(x, name='Relu_2')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    x = tf.layers.conv2d(inputs=x, name='conv_3', padding='same', filters=64, kernel_size=3, activation=None)
    x = tf.layers.batch_normalization(inputs=x, axis=-1, center=True, scale=True, name='batch_norm_3')
    x = tf.nn.relu(x, name='Relu_3')


    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

    x = tf.layers.flatten(x)


    x = tf.layers.dense(inputs=x, name='fc1', units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc2', units=64, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc3', units=num_classes, activation=None)


    return x, tf.nn.softmax(logits=x)



def residual_blocks(input, filters, kernel_size):

    y = tf.layers.conv2d(inputs=input, padding='same', filters=filters, kernel_size=kernel_size, activation=None, strides=1)
    y = tf.layers.batch_normalization(y, axis=-1, center=True, scale=True)

    x = tf.layers.conv2d(inputs=input, padding='same', filters=filters, kernel_size=kernel_size, activation=None)
    x = tf.layers.batch_normalization(inputs=x, axis=-1, center=True, scale=True)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, padding='same', filters=filters, kernel_size=kernel_size, activation=None)
    x = tf.layers.batch_normalization(inputs=x, axis=-1, center=True, scale=True)
    x = tf.nn.relu(x)

    x = tf.math.add(x, y)

    x = tf.nn.relu(x)

    return x


def ResNetModel(Img, ImageSize, MiniBatchSize):
    x = tf.layers.conv2d(inputs=Img, name='conv_1', padding='same', filters=16, kernel_size=3, activation=None)
    x = tf.layers.batch_normalization(inputs=x, axis=-1, center=True, scale=True, name='batch_norm_1')
    x = tf.nn.relu(x,name='Relu_1')

    x = residual_blocks(x, 32, 5)
    x = residual_blocks(x, 64, 5)
    x = residual_blocks(x, 64, 5)

    x = tf.layers.average_pooling2d(inputs=x, pool_size=[np.shape(x)[1], np.shape(x)[2]], strides=1)
    x = tf.contrib.layers.flatten(x)

    x = tf.layers.dense(inputs=x, name='fc1', units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc2', units=128, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, name='fc3', units=10, activation=None)

    prLogits = x
    prSoftMax = tf.nn.softmax(x)

    return prLogits, prSoftMax


def denseBlock(Img, depth, filters, kernel_size):

    layer_outputs = []
    img = tf.layers.conv2d(inputs = Img, padding='same',filters = filters, kernel_size = kernel_size, activation = None)
    layer_outputs.append(img)

    for z in range(depth):
        img = tf.nn.relu(Img)
        img = tf.layers.conv2d(inputs = img, padding='same',filters = filters, kernel_size = kernel_size, activation = None)
        layer_outputs.append(img)
        net = tf.layers.conv2d(inputs=tf.concat(layer_outputs,axis=3), padding='same',filters = filters, kernel_size = kernel_size, activation = None)

    return net

def DenseNetModel(Img, ImageSize, MiniBatchSize, num_classes = 10):


    x = tf.layers.conv2d(Img,filters = 16,kernel_size = 5,activation = None)

    x = denseBlock(Img = x, depth = 4,filters = 16,kernel_size=5)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs = x, name ='fc1', units = 256, activation = tf.nn.relu)
    x = tf.layers.dense(inputs = x, name ='fc2',units=128, activation=tf.nn.relu)
    x = tf.layers.dense(inputs = x, name ='fc3', units = num_classes, activation = None)

    prLogits = x
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax





def ResNext_block(Img, cardinality,filters, kernel_size, resfilters1, res_kernel_size1, resfilters2, res_kernel_size2):
    x = Img
    x = tf.layers.conv2d(inputs = x, padding='same',filters = filters, kernel_size = kernel_size, activation = None)
    x = tf.layers.batch_normalization(inputs = x,axis = -1, center = True, scale = True)
    I_store = x

    branches = []
    for i in range(cardinality):

        b = tf.layers.conv2d(inputs=x, padding='same', filters=resfilters1, kernel_size=res_kernel_size1, activation=None)
        b = tf.layers.batch_normalization(inputs=b, axis=-1, center=True, scale=True)
        b = tf.nn.relu(b)

        b = tf.layers.conv2d(inputs=b, padding='same', filters=resfilters2, kernel_size=res_kernel_size2, activation=None)
        b = tf.layers.batch_normalization(inputs=b, axis=-1, center=True, scale=True)

        branches.append(b)

    x = tf.concat(branches, axis = 3)

    x = tf.layers.conv2d(inputs = x, padding='same',filters = filters, kernel_size = kernel_size, activation = None)
    x = tf.layers.batch_normalization(inputs = x,axis = -1, center = True, scale = True)

    x = tf.math.add(x, I_store)

    x = tf.nn.relu(x)

    return x



def ResNextModel(Img, ImageSize, MiniBatchSize, num_classes=10):

    x = ResNext_block(Img = Img, cardinality = 5,filters = 16, kernel_size = 3, resfilters1 = 16, res_kernel_size1= 1, resfilters2 = 32, res_kernel_size2= 5)

    x = tf.layers.flatten(x)

    x = tf.layers.dense(inputs = x, name = 'fc1', units = 128, activation = tf.nn.relu)
    x = tf.layers.dense(inputs = x, name = 'fc2',units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs = x, name = 'fc3', units = num_classes, activation = None)


    prLogits = x
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax






