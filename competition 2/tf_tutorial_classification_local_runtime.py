# loading data from google drive into Colaboratory
#from google.colab import drive
#drive.mount('/content/drive')

import argparse
import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main(argv):

    # args setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=150)
    parser.add_argument("--train_dir", type=str, default=os.environ["HOME"])
    args = parser.parse_args(argv)
    batch_size = args.batch_size
    img_size = args.img_size
    train_dir = args.train_dir

    epochs = 5 #was originally 15. after using .2 of dataset for validation, converged before 5 epochs
    IMG_HEIGHT = img_size
    IMG_WIDTH = img_size
    class_mode = 'categorical'
    classes = ["00", "01", "02"]
    validation_split = 0.2

    image_generator = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    train_data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode=class_mode,
                                                            classes=classes,
                                                            subset='training')

    val_data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=train_dir,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode=class_mode,
                                                            classes=classes,
                                                            subset='validation')

    steps_per_epoch = train_data_gen.samples / batch_size
    validation_steps_per_epoch = val_data_gen.samples / batch_size

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=int(steps_per_epoch),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=int(validation_steps_per_epoch)
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    augmented_image_generator = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=45,
                        width_shift_range=.15,
                        height_shift_range=.15,
                        horizontal_flip=True,
                        zoom_range=0.5,
                        validation_split=validation_split)

    train_data_gen = augmented_image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode=class_mode,
                                                        classes=classes,
                                                        subset='training')

    val_data_gen = augmented_image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=train_dir,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode=class_mode,
                                                    classes=classes,
                                                    subset='validation')

    model_new = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
            input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model_new.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    model_new.summary()

    history = model_new.fit_generator(
        train_data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=validation_steps_per_epoch
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])

# eof
