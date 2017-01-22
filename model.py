import scipy.misc
import time
import os

import numpy as np
import tensorflow as tf


class VAE(object):

  def __init__(self, 
               sess,
               input_data=None,
               batch_size=100,
               checkpoint_dir=None):

    self.sess = sess
    self.input_data = input_data
    self.batch_size = batch_size

    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    self.x = tf.placeholder(tf.float32, shape=[None, 784])

    self.weights = {
      'enc_w1': tf.Variable(tf.random_normal([784, 1000], stddev=.1), name='enc_w1'),
      'enc_w2': tf.Variable(tf.random_normal([1000, 500], stddev=.1), name='enc_w2'),
      'enc_w3': tf.Variable(tf.random_normal([500, 250], stddev=.1), name='enc_w3'),
      'mu_w' : tf.Variable(tf.random_normal([250, 30], stddev=.1), name='mu_w'),
      'logsd_w': tf.Variable(tf.random_normal([250, 30], stddev=.1), name='logsd_w'),
      'dec_w1': tf.Variable(tf.random_normal([30, 250], stddev=.1), name='dec_w1'),
      'dec_w2': tf.Variable(tf.random_normal([250, 500], stddev=.1), name='dec_w2'),
      'dec_w3': tf.Variable(tf.random_normal([500, 1000], stddev=.1), name='dec_w3'),
      'dec_w4': tf.Variable(tf.random_normal([1000, 784], stddev=.1), name='dec_w4')
    }

    self.biases = {
      'enc_b1': tf.Variable(tf.zeros([1000]), name='enc_b1'),
      'enc_b2': tf.Variable(tf.zeros([500]), name='enc_b2'),
      'enc_b3': tf.Variable(tf.zeros([250]), name='enc_b3'),
      'mu_b': tf.Variable(tf.zeros([30]), name='mu_b'),
      'logsd_b': tf.Variable(tf.zeros([30]), name='logsd_b'),
      'dec_b1': tf.Variable(tf.zeros([250]), name='dec_b1'),
      'dec_b2': tf.Variable(tf.zeros([500]), name='dec_b2'),
      'dec_b3': tf.Variable(tf.zeros([1000]), name='dec_b3'),
      'dec_b4': tf.Variable(tf.zeros([784]), name='dec_b4')
    }

    self.pred, self.loss = self.model()

    self.saver = tf.train.Saver()

  def model(self):
    # Hidden layer encoder
    with tf.variable_scope("enc1"):
      enc1 = tf.nn.relu(tf.matmul(self.x, self.weights['enc_w1']) + self.biases['enc_b1'])
    with tf.variable_scope("enc2"):
      enc2 = tf.nn.relu(tf.matmul(enc1, self.weights['enc_w2']) + self.biases['enc_b2'])
    with tf.variable_scope("enc3"):
      enc3 = tf.nn.relu(tf.matmul(enc2, self.weights['enc_w3']) + self.biases['enc_b3'])

    # Mu encoder
    with tf.variable_scope("enc_mu"):
      enc_mu = tf.matmul(enc3, self.weights['mu_w']) + self.biases['mu_b']

    # Sigma encoder
    with tf.variable_scope("enc_logsd"):
      enc_logsd = tf.matmul(enc3, self.weights['logsd_w']) + self.biases['logsd_b']

    # Sample epsilon
    epsilon = tf.random_normal(tf.shape(enc_logsd), name='epsilon') # [?, 30]

    # Sample latent variable
    std_encoder = tf.exp(.5 * enc_logsd) # [?, 30]    

    # Compute KL divergence (latent loss)
    KLD = -.5 * tf.reduce_sum(1. + enc_logsd - tf.pow(enc_mu, 2) - tf.exp(enc_logsd), reduction_indices=1)

    # Generate z
    # z = mu + (sigma * epsilon)
    z = enc_mu + tf.mul(std_encoder, epsilon)

    # Hidden layer decoder
    with tf.variable_scope("dec1"):
      dec1 = tf.nn.relu(tf.matmul(z, self.weights['dec_w1']) + self.biases['dec_b1'])
    with tf.variable_scope("dec2"):
      dec2 = tf.nn.relu(tf.matmul(dec1, self.weights['dec_w2']) + self.biases['dec_b2'])
    with tf.variable_scope("dec3"):
      dec3 = tf.nn.relu(tf.matmul(dec2, self.weights['dec_w3']) + self.biases['dec_b3'])
    with tf.variable_scope("dec4"):
      x_hat = tf.matmul(dec3, self.weights['dec_w4']) + self.biases['dec_b4']

    # Compute binary cross entropy (recontruction loss)
    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, self.x), reduction_indices=1)

    # Compute loss
    loss = tf.reduce_mean(KLD + BCE)

    #     # Compute regularized loss
    #     regularized_loss = loss + FLAGS.lam * l2_loss

    return x_hat, loss

  def train(self, config):
    self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

    tf.initialize_all_variables().run()

    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    print("Start training...")

    for ep in xrange(config.epoch):
      for step in xrange(config.training_step):
        batch = self.input_data.train.next_batch(config.batch_size)

        counter += 1
        _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.x: batch[0]})

        if counter % 50 == 0:
          print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
            % ((ep+1), counter, time.time()-start_time, err))

        if (ep + 1) % 10 == 0 and counter % 9990 == 0:
          self.save(config.checkpoint_dir, counter)
          result = self.sess.run(self.pred, feed_dict={self.x: batch[0]})
          test_img = np.reshape(result[0], (28, 28))
          image_model = "{}_{}.png".format("test_img", counter)
          scipy.misc.imsave(os.path.join(os.getcwd(), image_model), test_img)

  def save(self, checkpoint_dir, step):
    model_name = "vae.model"
    model_dir = "{}".format("vae")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, 
                    os.path.join(checkpoint_dir, model_name), 
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "{}".format("vae")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False
