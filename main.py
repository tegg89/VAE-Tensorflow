from model import VAE

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Number of epochs [100]")
flags.DEFINE_integer("training_step", 10000, "Number of training steps [10000]")
flags.DEFINE_integer("batch_size", 100, "The size of batch sizes [100]")
flags.DEFINE_float("learning_rate", 3e-4, "The learning rate of optimizing algorithm [0.0003]")
flags.DEFINE_integer("lam", .01, "Lambda regularizer [0.01]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Checkpoint directory [checkpoint_dir]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  mnist = input_data.read_data_sets('MNIST')

  with tf.Session() as sess:
    vae = VAE(sess, 
              input_data=mnist, 
              batch_size=FLAGS.batch_size, 
              checkpoint_dir=FLAGS.checkpoint_dir)
  
    vae.train(FLAGS)

if __name__ == '__main__':
  tf.app.run()
