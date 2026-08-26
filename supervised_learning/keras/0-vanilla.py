#!/usr/bin/env python3
"""
Module to create a vanilla autoencoder using Keras.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder network.

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer
                               in the encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Build Encoder
    input_encoder = keras.Input(shape=(input_dims,))
    x = input_encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent_output = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=input_encoder, outputs=latent_output)

    # Build Decoder
    input_decoder = keras.Input(shape=(latent_dims,))
    x = input_decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output_decoder = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=input_decoder, outputs=output_decoder)

    # Build Full Autoencoder
    input_auto = keras.Input(shape=(input_dims,))
    encoded = encoder(input_auto)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=input_auto, outputs=decoded)

    # Compile Autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
