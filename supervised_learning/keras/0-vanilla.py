#!/usr/bin/env python3
"""
Module to build a vanilla autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder network.

    Parameters:
    - input_dims: integer, dimensions of the model input
    - hidden_layers: list of integers, number of nodes for each hidden layer
                     in the encoder
    - latent_dims: integer, dimensions of the latent space representation

    Returns:
    - encoder: the encoder model
    - decoder: the decoder model
    - auto: the full autoencoder model
    """
    # Build Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent_output = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=encoder_input, outputs=latent_output)

    # Build Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoded_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_input, outputs=decoded_output)

    # Build Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded_repr = encoder(auto_input)
    auto_output = decoder(encoded_repr)
    auto = keras.Model(inputs=auto_input, outputs=auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
