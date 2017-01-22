# VAE-Tensorflow
Implemented variational auto-encoder with MNIST dataset using TensorFlow.

## Prerequisites
  * Tensorflow
  * Scipy

## Usage
`python main.py`
Training process is activated, as parameters would be saved as checkpoint file. After every 10 epochs, generated MNIST number will be saved as png file.

## References
* [y0ast/VAE-TensorFlow](https://github.com/y0ast/VAE-TensorFlow)
  * I got intuition of variational auto-encoder implementation in this repository.
* [cdoersch/vae-tutorial](https://github.com/cdoersch/vae-tutorial)
  * This repository is supplemented code to the [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908). This code is Caffe version of variational auto-encoder for MNIST dataset.
