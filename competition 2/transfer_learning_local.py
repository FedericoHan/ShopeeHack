# -*- coding: utf-8 -*-
"""transfer_learning-local.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JvezlAycfxRkY1wo6GLmk19LXOuBRimE
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
from tensorflow.keras import layers

import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

#for local env
PATH = pathlib.Path('/Users/k/Documents/GitHub/shopee-local/shopee-product-detection-dataset')
PATH2 = pathlib.Path('/Users/k/Documents/GitHub/ShopeeHack/competition 2')
train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH,'testdir')
classifier_dir = os.path.join(PATH,'imagenet_resnet_v2_50_classification_4')
feature_extractor_dir = os.path.join(PATH,'imagenet_resnet_v2_50_feature_vector_4')

#for colab
#train_dir = ("/content/drive/My Drive/shopee2/train")

batch_size = 128
epochs = 5 #was originally 15. after using .2 of dataset for validation, converged before 5 epochs
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMAGE_SHAPE = (150, 150)
class_mode = 'categorical'
test_batch_size = 1 #must be one to get answer for each image
test_class_mode = None
#classes = ["00", "01", "02"]
validation_split = 0.2

augmented_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5,
    validation_split=validation_split)

train_data_gen = augmented_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode=class_mode,
    #classes=classes,
    subset='training')

#val_data_gen = augmented_image_generator.flow_from_directory(
#   batch_size=batch_size,
#    directory=train_dir,
#    target_size=(IMG_HEIGHT, IMG_WIDTH),
#    class_mode=class_mode,
    #classes=classes,
#    subset='validation')

image_generator = ImageDataGenerator(rescale=1./255)
test_data_gen = image_generator.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=test_batch_size,
    class_mode=test_class_mode,
    shuffle=False)

#WHAT DOES THIS NEED TO DO?
classifier_ex_layer = hub.KerasLayer(
    classifier_dir,
    input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
classifier = tf.keras.Sequential([
    classifier_ex_layer,
    layers.Dense(train_data_gen.num_classes, activation='softmax')
])
classifier.build([None, IMG_HEIGHT, IMG_WIDTH, 3])
classifier.summary()
#classifier_result = classifier.predict(test_data_gen)

test_steps_per_epoch = test_data_gen.samples // test_data_gen.batch_size
class_pred = classifier.predict_generator(
    test_data_gen,
    steps=test_steps_per_epoch,
    verbose=1)

predicted_class_indices=np.argmax(class_pred,axis=1)
labels = (train_data_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_data_gen.filenames
#FILENAME AND PREDICTIONS ARRAYS ARE DIFFERENT LENGTHS with validation data
results=pd.DataFrame({"filename":filenames, "category":predictions})
results["filename"] = results["filename"].str.replace("test/","")
import time
t = time.time()
results_filename = "class_results" +str(t)+".csv"
results.to_csv(results_filename, index=False)

feature_ex_layer = hub.KerasLayer(
    feature_extractor_dir, 
    input_shape=(IMG_HEIGHT,IMG_WIDTH,3),
    trainable=False)

model = tf.keras.Sequential([
  feature_ex_layer,
  layers.Dense(train_data_gen.num_classes, activation='softmax')
])
model.build([None, IMG_HEIGHT, IMG_WIDTH, 3])
model.summary()

#ONCE RETRAINED MODEL IS DONE, PASS THROUGH HERE TO RUN PREDICTION AND CREATE RESULTS.CSV

#THIS MODEL RETURNS ALL CAT 00
#model_saved = tf.keras.models.load_model(os.path.join(PATH2,'all_categories_all_augs.h5'))

## using data_generator.flow_from_directory()
test_steps_per_epoch = test_data_gen.samples // test_data_gen.batch_size
pred = model.predict_generator(
    test_data_gen,
    steps=test_steps_per_epoch,
    verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_data_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_data_gen.filenames
#FILENAME AND PREDICTIONS ARRAYS ARE DIFFERENT LENGTHS with validation data
results=pd.DataFrame({"filename":filenames, "category":predictions})
results["filename"] = results["filename"].str.replace("test/","")
import time
t = time.time()
results_filename = "results" +str(t)+".csv"
results.to_csv(results_filename, index=False)