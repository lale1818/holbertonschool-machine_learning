# Autoencoders

This project covers the implementation of various autoencoder architectures using TensorFlow/Keras, including vanilla, sparse, convolutional, and variational autoencoders.

## Tasks

### 0. "Vanilla" Autoencoder
`0-vanilla.py` contains a function `autoencoder(input_dims, hidden_layers, latent_dims)` that builds a basic (vanilla) autoencoder:
- The encoder compresses the input through a series of Dense layers (ReLU activations) down to a latent-space representation.
- The decoder mirrors the encoder's hidden layers in reverse to reconstruct the input, with a final sigmoid activation.
- The full autoencoder model is compiled with the Adam optimizer and binary cross-entropy loss.

## Requirements
- Ubuntu 20.04 LTS
- Python 3.9
- TensorFlow (Keras)
- pycodestyle style guidelines
