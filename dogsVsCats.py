import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', warn=False, force=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb
import os
import random
import gc

# returns x, a list of resized images, and y, a list of labels for those images
def readAndProcessImages(listOfImages, nrows, ncols):
    x = []  # images
    y = []  # labels
    for img in listOfImages:
        x.append(cv2.resize(cv2.imread(img, cv2.IMREAD_COLOR), (nrows, ncols), interpolation=cv2.INTER_CUBIC))
        if 'dog' in img:
            y.append(1)
        elif 'cat' in img:
            y.append(0)
    return x, y

def setup():
    trainDir = 'train/'
    testDir = 'test/'
    # We would use the following two lines if we wanted to only use a subset of our dataset which would require 
    # splitting it into cats and dogs before taking the first 2000 of each.
    #trainDogs = ['train/{}'.format(i) for i in os.listdir(trainDir) if 'dog' in i]  #get dog images 
    #trainCats = ['train/{}'.format(i) for i in os.listdir(trainDir) if 'cat' in i]  #get cat images

    #print('Number of dog training images: {0}.'.format(len(trainDogs)))
    #print('Number of cat training images: {0}.'.format(len(trainCats)))

    testImgs = ['test/{}'.format(i) for i in os.listdir(testDir)] # get test images

    # Again, if we were using a subset of the training images
    #trainImgs = trainDogs[:2000] + trainCats[:2000]
    trainImgs = ['train/{}'.format(i) for i in os.listdir(trainDir)]  #get training images 
    random.shuffle(trainImgs) # randomly shuffle the array of training images

    gc.collect()    # This would be more important if we used intermediate lists

    #for ima in trainImgs[0:3]:
    #    img = mpimg.imread(ima)
    #    imgplot = plt.imshow(img)
    #    plt.show()

    # Resize images using cv2
    nrows = 150
    ncols = 150
    channels = 3

    x, y = readAndProcessImages(trainImgs, nrows, ncols)
    
    del trainImgs
    gc.collect()

    # convert list to numpy array
    x = np.array(x)
    y = np.array(y)

    # Plot the labels to make sure we have two classes
    sb.countplot(y)
    plt.title('Labels for Cats and Dogs')

setup()
