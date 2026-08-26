#!/usr/bin/env python3
"""
Wasserstein GAN with Weight Clipping module
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
    WGAN with weight clipping class inheriting from keras.Model
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
    ):
        """
        Class constructor for WGAN_clip
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define the discriminator loss and optimizer:
        self.discriminator.loss = (
            lambda x, y: tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generates fake samples using the generator
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Generates real samples from real_examples
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Performs one training step of the WGAN
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                real_output = self.discriminator(
                    real_samples, training=True
                )
                fake_output = self.discriminator(
                    fake_samples, training=True
                )

                discr_loss = self.discriminator.loss(
                    fake_output, real_output
                )

            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1, 1))

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(
                fake_samples, training=True
            )
            gen_loss = self.generator.loss(fake_output)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
