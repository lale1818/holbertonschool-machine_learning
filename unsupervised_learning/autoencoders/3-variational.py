#!/usr/bin/env python3
"""
Variational Autoencoder module.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for hidden layers in encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): Encoder model outputting [z, z_mean, z_log_var]
        decoder (keras.Model): Decoder model
        auto (keras.Model): Full autoencoder model
    """
    # -------------------
    # 1. ENCODER
    # -------------------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mean, log_var = args
        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.int_shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mean + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )
    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # -------------------
    # 2. DECODER
    # -------------------
    latent_inputs = keras.Input(shape=(latent_dims,))
    y = latent_inputs
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(nodes, activation='relu')(y)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(y)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # -------------------
    # 3. FULL AUTOENCODER
    # -------------------
    z_sampled, mean, log_var = encoder(inputs)
    auto_outputs = decoder(z_sampled)
    auto = keras.Model(inputs, auto_outputs, name='auto')

    # Calculate VAE loss (Reconstruction Loss + KL Divergence)
    def vae_loss(x, x_decoded_mean):
        reconstruction_loss = keras.losses.binary_crossentropy(
            x, x_decoded_mean
        )
        reconstruction_loss *= input_dims
        kl_loss = 1 + log_var - keras.backend.square(mean) - \
            keras.backend.exp(log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
