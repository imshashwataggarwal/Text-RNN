from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from helper import *

class text_rnn(object):
    def __init__(self, batch_size, seq_length, rnn_size, num_layers, vocab_size, grad_clip=5, training=True):
        super(text_rnn, self).__init__()

        if training == False:
            batch_size, seq_length = 1, 1

        tf.reset_default_graph()

        self.get_inputs(batch_size, seq_length)
        self.get_rnn(batch_size, rnn_size, num_layers)
        self.get_output(vocab_size)
        self.build_nn(rnn_size, vocab_size)
        self.probs = tf.nn.softmax(tf.divide(self.logits, self.temp), name='probs')
        self.get_loss(rnn_size, vocab_size)
        self.get_optimizer(grad_clip)

    def get_inputs(self, batch_size, seq_length):

        self.inputs = tf.placeholder(tf.int32, [batch_size,seq_length], name='inputs')
        self.targets = tf.placeholder(tf.int32,[batch_size,seq_length], name='targets')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.temp = tf.placeholder(tf.float32, name='temp')

    def get_rnn(self, batch_size, rnn_size, num_layers):

        def lstm_cell(rnn_size, keep_prob):
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size, self.keep_prob) for _ in range(num_layers)])
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

    def get_output(self, vocab_size):

        inputs_one_hot = tf.one_hot(self.inputs, vocab_size)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, inputs_one_hot, initial_state=self.initial_state)

    def build_nn(self, rnn_size, vocab_size):

        rnn_output = tf.concat(self.outputs, axis=1)
        rnn_output = tf.reshape(rnn_output, [-1, rnn_size])
        self.logits = tf.layers.dense(rnn_output, vocab_size)

    def get_loss(self, rnn_size, vocab_size):
        labels = tf.one_hot(self.targets, vocab_size)
        labels = tf.reshape(labels, self.logits.get_shape())
        # Softmax cross entropy loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels))

    def get_optimizer(self, grad_clip):

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))
