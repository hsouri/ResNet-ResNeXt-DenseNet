#!/usr/bin/env python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision


Author(s):
Hossein Souri (hsouri@terpmail.umd.edu)
PhD Candidate in Electrical and Computer Engineering,
Computer Vision and Machine Learining
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import SimpleModel, ImprovedModel, ResNetModel, DenseNetModel, ResNextModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import itertools


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    # I1S = iu.StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1, axis=0)

    return I1Combined, I1
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, prSoftMaxS = ImprovedModel(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        OutSaveT = open(LabelsPathPred, 'w')

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

            OutSaveT.write(str(PredT)+'\n')
            
        OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = list(map(float, LabelTest.split()))

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = list(map(float, LabelPred.split()))
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred, ExpName):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')

    plot_confusion_matrix(cm, name=ExpName)


def plot_confusion_matrix(confusion_matrix, normalize=True, name='ImprovedModel', title='Confusion matrix', cmap=plt.cm.Blues, num_of_classes=10):
    if normalize:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(name + ' ' + title)
    plt.colorbar()
    tick_marks = np.arange(num_of_classes)
    plt.xticks(tick_marks, np.arange(10))
    plt.yticks(tick_marks, np.arange(10))

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name + '_confusion_matrix.jpg')

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ExpName', dest='ExpName', default='ImprovedModel', help='Name of Experiment')
    Parser.add_argument('--NumEpochs', dest='NumEpochs', type=int, default=24, help='Path to load images from, Default:BasePath')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/scratch0/CMSC733/YourDirectoryID_hw0/Phase2/Checkpoints/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/scratch0/CMSC733/YourDirectoryID_hw0/Phase2/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ExpName = Args.ExpName
    NumEpochs = Args.NumEpochs
    ModelPath_base = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    test_acc = []
    for epoch in range(NumEpochs):
        # Parse Command Line arguments
        tf.reset_default_graph()

        ModelPath = ModelPath_base + str(epoch) + 'model.ckpt'
        print(ModelPath)


        # Setup all needed parameters including file reading
        ImageSize, DataPath = SetupAll(BasePath)

        # Define PlaceHolder variables for Input and Predicted output
        ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
        LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

        TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred)

        # Plot Confusion Matrix
        LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
        accuracy = Accuracy(LabelsTrue, LabelsPred)
        test_acc.append(accuracy)
        np.save(ExpName + '_test_accuracy.npy', np.array(test_acc))
        print(test_acc)


    ConfusionMatrix(LabelsTrue, LabelsPred, ExpName)

     
if __name__ == '__main__':
    main()
