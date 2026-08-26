#!/usr/bin/env python3
"""Defines a function that creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder

    Args:
        input_dims: tuple of integers containing the dimensions of the
            model input
        filters: list containing the number of filters for each
            convolutional layer in the encoder, respectively. The filters
            should be reversed for the decoder
        latent_dims: tuple of integers containing the dimensions of the
            latent space representation

    Returns:
        encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(encoder_inputs, x)

    # Decoder
    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs
    reversed_filters = filters[::-1]
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    # Second to last convolution uses valid padding
    x = keras.layers.Conv2D(
        reversed_filters[-1], (3, 3), padding='valid', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Last convolution matches the number of input channels, no upsampling
    decoder_outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
