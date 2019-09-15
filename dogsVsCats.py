import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg', warn=False, force=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb

from sklearn.model_selection import train_test_split

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

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
    #sb.countplot(y)
    #plt.title('Labels for Cats and Dogs')

    # Use 20% of the training data for a validation set and the other 80% for training
    xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=0.20, random_state=2)
   
    print('Shape of training images is {0}'.format(xTrain.shape))
    print('Shape of validation images is {0}'.format(xVal.shape))
    print('Shape of training labels is {0}'.format(yTrain.shape))
    print('Shape of validation labels is {0}'.format(yVal.shape))

    del x
    del y
    gc.collect()

    # get the size of the train and validation data
    ntrain = len(xTrain)
    nval = len(xVal)

    # Batch size should be a power of 2. We will use 32
    batchSize = 32

    model = models.Sequential()
    model.add(layers.Conv2d(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2d(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2d(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2d(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid')) # Sigmoid at the end because we have only two classes

    # We will use the RMSprop optimizer with a learning rate of 0.0001
    # Since we are doing a binary classification, we will use binary crossentropy loss
    model.compile(loss='binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics=['acc'])
    
    # Create an augmentation configuration
    # This helps prevent overfitting since we are using a small dataset
    trainDatagen = ImageDataGenerator(  rescale = 1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip = True,)
    valDatagen = ImageDataGenerator(rescale = 1./255)   # We do not augment the validation data, just rescale it

    # Use the above to create the actual image generators
    trainGenerator = trainDatagen.flow(xTrain, yTrain, batch_size=batchSize)
    valGenerator = valDatagen.flow(xVal, yVal, batch_size=batchSize)

    # Now we train for 64 epochs with about 100 steps per epoch
    history = model.fit_generator(  trainGenerator,
                                    steps_per_epoch = ntrain // batchSize,
                                    epochs = 64,
                                    validation_data = valGenerator,
                                    validation_steps = nval / batchSize)

    model.save_weights('modelWeights.h5')
    model.save('modelKeras.h5')

    return history

def plotResults(history):
    # Plot the training and validation curves
    acc = history.history['acc']
    valAcc = history.history['val_acc']
    loss = history.history['loss']
    valLoss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Training and validation accuracies
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, valAcc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.figure()

    # Training and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, valLoss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

   plt.show() 

def predict(testSize, testSet):
    xTest, yTest = read_and_process_image(testSet[0:testSize]) # yTest should be empty
    x = np.array(xTest)
    testDatagen = ImageDataGenerator(rescale = 1./255) # Like the validation set, we do not augment the test set

    ndx = 0;
    textLabels = []
    plt.figure(figsize=(30, 20))
    for batch in testDatagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        if pred > 0.5:
            textLabels.append('Dog')
        else:
            textLabels.append('Cat')
        plt.subplot(5 / columns + 1, columns, ndx + 1)
        plt.title('This is a {0}'.format(textLabels[ndx]))
        imgplot = plt.imshow(batch[0])
        ndx += 1
        if ndx % 10 == 0:
            break
    plt.show()

def run():
    history = setup()
    plotResults(history)
