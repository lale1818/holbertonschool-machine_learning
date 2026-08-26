#!/usr/bin/env python3
"""
Convolutional Generator and Discriminator module
"""
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds a convolutional generator and discriminator
    for generating 16x16 faces

    Returns:
        generator (keras.Model): Generator model
        discriminator (keras.Model): Discriminator model
    """

    def get_generator():
        """Builds generator model"""
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(16,)),
                keras.layers.Dense(2048),
                keras.layers.Reshape((2, 2, 512)),
                keras.layers.UpSampling2D(),
                keras.layers.Conv2D(64, (3, 3), padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("tanh"),
                keras.layers.UpSampling2D(),
                keras.layers.Conv2D(16, (3, 3), padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("tanh"),
                keras.layers.UpSampling2D(),
                keras.layers.Conv2D(1, (3, 3), padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("tanh"),
            ],
            name="generator",
        )
        return model

    def get_discriminator():
        """Builds discriminator model"""
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(16, 16, 1)),
                keras.layers.Conv2D(32, (3, 3), padding="same"),
                keras.layers.MaxPooling2D(),
                keras.layers.Activation("tanh"),
                keras.layers.Conv2D(64, (3, 3), padding="same"),
                keras.layers.MaxPooling2D(),
                keras.layers.Activation("tanh"),
                keras.layers.Conv2D(128, (3, 3), padding="same"),
                keras.layers.MaxPooling2D(),
                keras.layers.Activation("tanh"),
                keras.layers.Conv2D(256, (3, 3), padding="same"),
                keras.layers.MaxPooling2D(),
                keras.layers.Activation("tanh"),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ],
            name="discriminator",
        )
        return model

    return get_generator(), get_discriminator()
