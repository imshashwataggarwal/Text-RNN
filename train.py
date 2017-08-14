from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from helper import *
from model import *
import os

script = 'dbz'

pickle_path = './model/' + script + '/' + script + '_preprocess.p'
save_dir = './model/' + script + '/' + script

pickle_path = os.path.join(pickle_path)
save_dir = os.path.join(save_dir)

print("Training Started...\n")
int_text, vocab_to_int, int_to_vocab = load_preprocess(pickle_path)

print("Data Loaded! \n")

vocab_size = len(vocab_to_int.keys())

#Hyperparameters.
num_epochs  = 20
batch_size  = 100
rnn_size    = 256
num_layers  = 2
embed_dim   = 256
seq_length  = 100
lr          = 0.001
keep_prob   = 0.5
temperature = 1.0
grad_clip   = 5
decay_rate  = 0.97

nn = text_rnn(batch_size, seq_length, rnn_size, num_layers, vocab_size, embed_dim, grad_clip)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3,keep_checkpoint_every_n_hours=0.5)

with tf.Session() as sess:

    check = tf.train.latest_checkpoint('./model/'+ script +'/')
    if check == None:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, check)
        print("MODEL RESTORED", check)

    for epoch in range(num_epochs+1):
        state = sess.run(nn.initial_state)
        lr = lr * (decay_rate ** epoch)
        for (x, y) in next_batch(int_text, batch_size, seq_length):
            feed = {
                nn.inputs: x,
                nn.targets: y,
                nn.initial_state: state,
                nn.learning_rate: lr,
                nn.keep_prob: keep_prob,
                nn.temp: temperature
                }
            train_loss, state, _ = sess.run([nn.cost, nn.final_state, nn.optimizer], feed_dict=feed)

        print('Epoch %i  train_loss = %0.3f'%(epoch,train_loss))
        if epoch%50 == 0 and epoch:
            # Save Model
            saver.save(sess, save_dir, global_step=epoch)
            print('Model Saved...')
    # Save Model
    saver.save(sess, save_dir, global_step=epoch)
    print('Model Trained and Saved')
