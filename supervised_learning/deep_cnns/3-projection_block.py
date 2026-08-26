#!/usr/bin/env python3
"""
Module to build a projection block for ResNet using Keras.
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning.

    Parameters:
    - A_prev: output tensor from the previous layer
    - filters: tuple or list containing [F11, F3, F12]
    - s: stride of the first convolution in both main and shortcut paths

    Returns:
    - Activated output tensor of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main Path
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=-1)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Shortcut Path
    conv_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    bn_shortcut = K.layers.BatchNormalization(axis=-1)(conv_shortcut)

    # Add & Activate
    add = K.layers.Add()([bn3, bn_shortcut])
    output = K.layers.Activation('relu')(add)

    return output
