#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For reproducibility
np.random.seed(42)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from keras.preprocessing.image import ImageDataGenerator 
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[28]:


# only going to add padding, as making verious change in art can actually revlea charactersics of another period
# adding padding because edges are importnat.
#but may that is difficult to learn with sketches on white paper.
# hmm...


# In[3]:


# csv vile containg the image number and the style that goes with it.
df = pd.read_csv("./image_names_style.csv")

# ImageDataGenerator can manipulte images, I did, the rescale 1./255 because this makes the values between 0 & 1
# 
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25) #and i split my data here into training and testing

train_gen = datagen.flow_from_dataframe(
        dataframe=df, #the dataframe i read in
        directory='./images', #folder containing images
        x_col="new_filename", #column in the file that is the image name
        y_col="style", #the style that goes with the image, which is what I want to predict
        target_size=(256, 256),
        batch_size=32,
        subset='training',
        color_mode='rgb') # class_mode='categorical' #try with color mode brayscale

val_gen = datagen.flow_from_dataframe(
        dataframe=df, #the dataframe i read in
        directory='./images', #folder containing images
        x_col="new_filename", #column in the file that is the image name
        y_col="style", #the style that goes with the image, which is what I want to predict
        target_size=(256, 256),
        batch_size=32,
        subset='validation',
        color_mode='rgb') # class_mode='categorical'


# ### Creating the model
# 
# - CNN's go through two phases, a convolutional phase and then a fully connected or dense phase.
# - My convolution phase will have 4 stages
# - each activation function in this phase will be relu,
# - Each kernel size will be 3x3 which will also be the kernel size in the MaxPooling layer.
# - Included padding in each layer because I feel that the edges of each image are important.

# In[ ]:


# Instantiate a CNN.
model = Sequential()

# 1st convolutional layer.
model.add(Conv2D(filters = 64,
                       kernel_size = (3,3),
                       activation = 'relu',
                       padding='same',
                       input_shape = (256,256,3))) # first input shape is the same that is in the target size w/ 3 for the rbg channels

# 1st pooling layer.
model.add(MaxPooling2D(pool_size = (3,3))) # dimensions of region of pooling

# 2nd convolutional layer
model.add(Conv2D(128,
                       kernel_size = (3,3),
                       padding='same',
                       activation = 'relu'))

# 2nd pooling layer.
model.add(MaxPooling2D(pool_size = (3,3)))

# 3rd convolutional layer
model.add(Conv2D(256,
                       kernel_size = (3,3),
                       padding='same',
                       activation = 'relu'))

# 3rd pooling layer.
model.add(MaxPooling2D(pool_size = (3,3)))

# 4th convolutional layer
model.add(Conv2D(512,
                       kernel_size = (3,3),
                       padding='same',
                       activation = 'relu'))

# 4th pooling layer.
model.add(MaxPooling2D(pool_size = (3,3)))


# Flattening before moving to the fully connected phase.
model.add(Flatten())


# ### Second phase of CNN
# - It is done after the flattening. This moves toa  dense/fully connected phase, where the final neurons will be the number of categories (3).
# - Using softmax activation in the final dense layer because it is a multi classification problem.
# - Including a 0.5 dropout between each dense layer to avoid over fitting.
# - Metric is accuracy because I want to mesaure how accuratly my model will predict

# In[4]:



# 1st densely-connected layer
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))

# 2nd densely-connected layer
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))

# Final dense layer w/ 3 neurons for the 3 categories.
model.add(Dense(3, activation = 'softmax'))

# Compile model
model.compile(loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])

# Fit model on training data
# history = model.fit_generator(train_generator,steps_per_epoch=train_steps, epochs=20,
#                               validation_data=validation_generator,validation_steps=validation_steps)


# ### Fitting the model

# In[ ]:


history = model.fit_generator(train_gen,
                          validation_data = val_gen,
                          epochs = 100,
                          verbose = 1)


# In[ ]:


#train_gen = X_train.reshape(X_train.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[6]:


model.summary()


# In[9]:


# from keras.utils.visualize_util import plot
# import pydot
# plot(model, show_shapes=True, to_file='model.png')


# In[ ]:




