import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import gc

trainDir = 'train/'
testDir = 'test/'
# We would use the following two lines if we wanted to only use a subset of our dataset which would require 
# splitting it into cats and dogs before taking the first 2000 of each.
#trainDogs = ['train/{}'.format(i) for i in os.listdir(trainDir) if 'dog' in i]  #get dog images 
#trainCats = ['train/{}'.format(i) for i in os.listdir(trainDir) if 'cat' in i]  #get cat images

print('Number of dog training images: {0}.'.format(len(trainDogs)))
print('Number of cat training images: {0}.'.format(len(trainCats)))

testImgs = ['test/{}'.format(i) for i in os.listdir(testDir)] # get test images

# Again, if we were using a subset of the training images
#trainImgs = trainDogs[:2000] + trainCats[:2000]
trainImgs = ['train/{}'.format(i) for i in os.listdir(trainDir)]  #get training images 
random.shuffle(trainImgs) # randomly shuffle the array of training images

