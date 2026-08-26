#!/usr/bin/env python3
"""Defines a function that creates a variational autoencoder"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
            layer in the encoder, respectively. The hidden layers should be
            reversed for the decoder
        latent_dims: integer containing the dimensions of the latent
            space representation

    Returns:
        encoder, decoder, auto
            encoder is the encoder model, which outputs the latent
                representation, the mean, and the log variance,
                respectively
            decoder is the decoder model
            auto is the full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Samples from the latent distribution using the
        reparameterization trick"""
        mean, log_sigma = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mean + K.exp(log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder = keras.Model(encoder_inputs, [z, z_mean, z_log_sigma])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(
        input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs)[0])
    auto = keras.Model(encoder_inputs, auto_outputs)

    def vae_loss(x, x_decoded):
        """Computes the VAE loss as reconstruction loss plus the KL
        divergence between the latent distribution and a standard
        normal distribution"""
        reconstruction_loss = keras.losses.binary_crossentropy(x, x_decoded)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
