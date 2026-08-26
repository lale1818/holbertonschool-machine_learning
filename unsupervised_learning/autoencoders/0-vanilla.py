#!/usr/bin/env python3
"""
Vanilla Autoencoder module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder model

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent_outputs = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_inputs, latent_outputs, name='encoder')

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    # Full Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, auto_outputs, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
