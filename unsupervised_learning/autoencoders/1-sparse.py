#!/usr/bin/env python3
"""Defines a function that creates a sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
            layer in the encoder, respectively. The hidden layers should be
            reversed for the decoder
        latent_dims: integer containing the dimensions of the latent
            space representation
        lambtha: regularization parameter used for L1 regularization on
            the encoded output

    Returns:
        encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the sparse autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    regularizer = keras.regularizers.l1(lambtha)
    latent = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=regularizer)(x)
    encoder = keras.Model(encoder_inputs, latent)

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(
        input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
