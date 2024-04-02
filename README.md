# SimpleVAEs

This repository contains two versions of the implementation of the variational autoencoder (simple/convolutional) in `tensorflow`. The implementations follow a modular design, allowing for easy modifications to the loss function and the architectures of the decoder/encoder. 

There is also an experimental implementation (deprecated) of Simple/cVAE in `jax` with `haiku`. 

You are recommended to use the `ResVAE` implementation, which is the best-performing implementation in this repo.

However, it has not been verified since the author lacks access to sufficient computational resources. The framework itself is original, but information from the following sources has been considered as references.

--- 

## Reference:

A mathematical survey: https://arxiv.org/abs/2101.00734

Implementation by DeepMind with `jax` and `haiku` of SimpleVAE: https://github.com/theorashid/jax-vae/blob/main/vae-classify.py

Official Implementation of cVAE with TensorFlow by Google: https://www.tensorflow.org/tutorials/generative/cvae
