#!/usr/bin/env python3
"""
Variational Autoencoder module.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for hidden layers in encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): Encoder model.
        decoder (keras.Model): Decoder model.
        auto (keras.Model): Full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mean, log_variance = args
        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.int_shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mean + keras.backend.exp(0.5 * log_variance) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )
    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    y = latent_inputs
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(nodes, activation='relu')(y)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(y)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Full Autoencoder
    z_sampled, _, _ = encoder(inputs)
    auto_outputs = decoder(z_sampled)
    auto = keras.Model(inputs, auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
