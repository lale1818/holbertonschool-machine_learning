#!/usr/bin/env python3
"""
Module to build an identity block for ResNet using Keras.
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning.

    Parameters:
    - A_prev: output tensor from the previous layer
    - filters: tuple or list containing [F11, F3, F12]

    Returns:
    - Activated output tensor of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # First component of main path: 1x1 Conv -> BN -> ReLU
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Second component of main path: 3x3 Conv -> BN -> ReLU
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=-1)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Third component of main path: 1x1 Conv -> BN
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Add shortcut value to main path, then apply ReLU
    add = K.layers.Add()([bn3, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
